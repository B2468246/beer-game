#!/usr/bin/env python3
"""Beer Game Classroom App — server with game engine, Claude AI, and WebSocket."""

import asyncio
import json
import math
import os
import random
import re
import socket
import string
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Global game state (in-memory, no DB) ─────────────────────────────────────

state = {
    "phase": "setup",  # setup | lobby | designing | playing | finished
    "session_id": None,
    "api_key": None,
    "settings": {
        "rounds": 10,
        "lead_time": 2,
        "initial_inventory": 12,
        "initial_pipeline": 4,
        "holding_cost": 0.50,
        "backlog_cost": 1.00,
        "demand_type": "step",       # "step" or "step_variance"
        "step_demand_before": 4,
        "step_demand_after": 8,
        "step_round": 5,             # demand changes at this round (1-indexed)
        "demand_std": 2,             # standard deviation (only used for step_variance)
        "warmup_rounds": 3,          # first N rounds don't count for scoring
        "max_games_per_player": 3,   # how many games each student may start
    },
    "players": {},       # {player_id: {name, roles:[], strategies:{}, locked_in:bool}}
    "teams": [],         # [{id, name, members:[], role_map:{role: player_id}}]
    "rounds_data": {},   # {team_id: {role: [{round, demand, inventory, backlog, pipeline, outstanding, order, cost, reasoning, incoming_shipment}]}}
    "current_round": 0,
    "demand_sequence": [],
    "created_at": None,
}

ws_clients: list[WebSocket] = []

ROLES = ["Retailer", "Wholesaler", "Distributor", "Manufacturer"]

# ── Persistence (JSON snapshot) ──────────────────────────────────────────────
# Render free tier has ephemeral storage; this still survives WebSocket drops
# and browser-close reconnects within the container lifetime.

STATE_FILE = os.environ.get("BEERGAME_STATE_FILE",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "game_state.json"))


def save_state() -> None:
    try:
        snapshot = {k: v for k, v in state.items() if k != "api_key"}
        # api_key kept separately so it's never written to disk
        with open(STATE_FILE, "w") as f:
            json.dump(snapshot, f)
    except Exception as e:
        print(f"[persist] save failed: {e}")


def load_state() -> None:
    if not os.path.exists(STATE_FILE):
        return
    try:
        with open(STATE_FILE) as f:
            snapshot = json.load(f)
        for k, v in snapshot.items():
            state[k] = v
        # Pick up API key from env on restart
        state["api_key"] = os.environ.get("ANTHROPIC_API_KEY", state.get("api_key"))
        # If server crashed mid-game, revert to designing so players can resume
        if state.get("phase") == "playing":
            state["phase"] = "designing"
        print(f"[persist] restored state: phase={state.get('phase')}, players={len(state.get('players', {}))}")
    except Exception as e:
        print(f"[persist] load failed: {e}")


# ── Demand generation ─────────────────────────────────────────────────────────

def generate_demand_sequence(settings: dict) -> list[int]:
    n = settings["rounds"]

    if settings["demand_type"] == "step_variance":
        mean = settings["step_demand_before"]
        std = settings.get("demand_std", 2)
        return [max(0, round(mean + random.gauss(0, std))) for _ in range(n)]

    # plain step-up
    before = settings["step_demand_before"]
    after = settings["step_demand_after"]
    switch = settings["step_round"] - 1  # 0-indexed
    return [before if i < switch else after for i in range(n)]


# ── Team assignment ───────────────────────────────────────────────────────────

def assign_teams(players: dict) -> list[dict]:
    ids = list(players.keys())
    random.shuffle(ids)
    n = len(ids)
    if n == 0:
        return []

    # Maximize teams of 3 and 4, avoid teams of 1 (unless n==1)
    if n == 1:
        teams_of_4, teams_of_3 = 0, 0
        remainder = 1
    elif n == 2:
        teams_of_4, teams_of_3 = 0, 0
        remainder = 2
    else:
        teams_of_4 = 0
        teams_of_3 = 0
        if n % 4 == 0:
            teams_of_4 = n // 4
        elif n % 4 == 1:
            teams_of_4 = (n // 4) - 1
            teams_of_3 = (n - teams_of_4 * 4) // 3
        elif n % 4 == 2:
            teams_of_4 = (n - 6) // 4 if n >= 6 else 0
            teams_of_3 = (n - teams_of_4 * 4) // 3 if n >= 6 else 0
            if n < 6:
                teams_of_3 = 0
                remainder = n
            else:
                remainder = n - teams_of_4 * 4 - teams_of_3 * 3
        elif n % 4 == 3:
            teams_of_4 = n // 4
            teams_of_3 = 1
        remainder = n - teams_of_4 * 4 - teams_of_3 * 3

    team_sizes = [4] * teams_of_4 + [3] * teams_of_3
    if remainder > 0 and remainder not in (0,):
        if remainder == 1 and len(team_sizes) == 0:
            team_sizes = [1]
        elif remainder == 2 and len(team_sizes) == 0:
            team_sizes = [2]
        elif remainder == 2:
            team_sizes.append(2)
        elif remainder == 1 and len(team_sizes) > 0:
            # shouldn't happen with logic above, but fallback
            team_sizes.append(1)

    teams = []
    idx = 0
    for i, size in enumerate(team_sizes):
        member_ids = ids[idx:idx + size]
        idx += size

        roles_shuffled = ROLES[:]
        random.shuffle(roles_shuffled)
        role_map = {}
        for j, pid in enumerate(member_ids):
            assigned = [roles_shuffled[j]]
            if size < 4:
                # distribute extra roles among members
                extra_roles = [r for r in ROLES if r not in [roles_shuffled[k] for k in range(size)]]
                per_member = len(extra_roles) // size
                leftover = len(extra_roles) % size
                start = j * per_member + min(j, leftover)
                end = start + per_member + (1 if j < leftover else 0)
                assigned += extra_roles[start:end]
            players[pid]["roles"] = assigned
            for r in assigned:
                role_map[r] = pid

        teams.append({
            "id": f"team_{i+1}",
            "name": f"Team {i+1}",
            "members": member_ids,
            "role_map": role_map,
        })

    return teams


# ── Base-Stock formula ────────────────────────────────────────────────────────

def base_stock_order(history: list[dict], lead_time: int) -> int:
    """S = mean_demand*(L+1) + z*sigma*sqrt(L+1); order = max(0, S - IP)"""
    demands = [h["demand"] for h in history]
    if not demands:
        return 4
    mean_d = sum(demands) / len(demands)
    sigma = (sum((d - mean_d) ** 2 for d in demands) / max(len(demands), 1)) ** 0.5
    z = 0.43
    L = lead_time
    S = mean_d * (L + 1) + z * sigma * math.sqrt(L + 1)

    last = history[-1]
    ip = last["inventory"] + last["pipeline"] + last["outstanding"] - last["backlog"]
    order = max(0, round(S - ip))
    return order


# ── Claude AI call ────────────────────────────────────────────────────────────

semaphore = asyncio.Semaphore(20)

SYSTEM_PROMPT_TEMPLATE = """You are one stage in a four-stage supply chain (Beer Distribution Game, {rounds} rounds):
Manufacturer -> Distributor -> Wholesaler -> Retailer -> End Customer.
Only the Retailer sees actual customer demand. All other stages see only the order from their immediate downstream stage.

Gameplay per week:
1. You receive goods from your supplier (incoming shipment from orders placed {lead_time} rounds ago).
2. Your customer places demand. You deliver from inventory. Unfilled demand = backlog.
3. You place your order with your supplier.

Rules:
- Lead time: {lead_time} weeks
- Pipeline: goods already shipped and on the way to you. Will arrive for sure.
- Outstanding: ordered but supplier hasn't shipped yet. Will be delivered - do NOT re-order.

{objective}

Parameters:
- Holding cost: {holding_cost:.2f} EUR per unit in inventory
- Backlog cost: {backlog_cost:.2f} EUR per unit (twice as expensive as holding)
- Already available = Inventory + Pipeline + Outstanding - Backlog

Respond with your ordering decision.
Write at the end: ORDER: <number>"""

OBJECTIVES = {
    "rational": "Objective: Minimize YOUR OWN cumulative costs only. Other stages are irrelevant to you - optimize exclusively your own inventory and backlog cost balance.",
    "cooperative": "Objective: Minimize the TOTAL COSTS OF ALL FOUR stages over all rounds. The chain is evaluated as a team - even if your own stage performs worse as a result.",
}


async def call_claude(api_key: str, system_prompt: str, user_message: str) -> tuple[int, str]:
    """Call Claude API, return (order_quantity, full_reasoning)."""
    async with semaphore:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": "claude-haiku-4-5-20251001",
            "max_tokens": 512,
            "temperature": 0.0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["content"][0]["text"]
                # Parse ORDER: <number>
                match = re.search(r"ORDER:\s*(\d+)", text)
                order = int(match.group(1)) if match else 4
                return order, text
            except Exception as e:
                return 4, f"[AI Error: {e}] Defaulting to order 4."


def build_user_message(role: str, round_num: int, rd: dict, history: list[dict]) -> str:
    lines = [
        f"Round {round_num} | Role: {role}",
        f"Inventory: {rd['inventory']} units",
        f"Pipeline (in transit): {rd['pipeline']} units",
        f"Outstanding (ordered, not shipped): {rd['outstanding']} units",
        f"Backlog: {rd['backlog']} units",
        f"Already available: {rd['inventory'] + rd['pipeline'] + rd['outstanding'] - rd['backlog']} units",
        f"Demand (this round): {rd['demand']} units",
        f"Incoming shipment (this round): {rd['incoming_shipment']} units",
        f"Cost so far: {rd['cumulative_cost']:.2f} EUR",
        "",
    ]
    if history:
        lines.append(f"Last {min(5, len(history))} rounds (oldest first):")
        for h in history[-5:]:
            lines.append(
                f"  Round {h['round']}: Demand={h['demand']}, Ordered={h['order']}, "
                f"Stock={h['inventory']}, Backlog={h['backlog']}, Pipeline={h['pipeline']}"
            )
        lines.append("")
    lines.append("How many units do you order? Write ORDER: <number>")
    return "\n".join(lines)


# ── Game engine ───────────────────────────────────────────────────────────────

def init_team_state(team: dict) -> dict:
    """Initialize round-by-round state trackers for a team."""
    lead_time = state["settings"]["lead_time"]
    init_inv = state["settings"]["initial_inventory"]
    init_pipe = state["settings"]["initial_pipeline"]

    team_state = {}
    for role in ROLES:
        team_state[role] = {
            "inventory": init_inv,
            "backlog": 0,
            "pipeline": [init_pipe] * lead_time,  # pipeline[0] arrives next round
            "outstanding": 0,
            "cumulative_cost": 0.0,
            "orders_placed": [],  # for bullwhip calculation
            "demands_received": [],
        }
    return team_state


async def run_game():
    """Main game loop — processes all teams round by round."""
    settings = state["settings"]
    n_rounds = settings["rounds"]
    lead_time = settings["lead_time"]
    demand_seq = state["demand_sequence"]

    # Initialize per-team live state
    team_states = {}
    for team in state["teams"]:
        team_states[team["id"]] = init_team_state(team)
        state["rounds_data"][team["id"]] = {role: [] for role in ROLES}

    for round_num in range(1, n_rounds + 1):
        state["current_round"] = round_num
        customer_demand = demand_seq[round_num - 1]

        # Tell clients the round is being computed (so they hide stale team
        # summary charts until the new numbers arrive).
        await broadcast({
            "type": "round_processing",
            "round": round_num,
            "total_rounds": n_rounds,
        })

        # Process all teams concurrently for this round
        tasks = []
        for team in state["teams"]:
            tasks.append(process_team_round(team, team_states[team["id"]], round_num, customer_demand, settings))
        await asyncio.gather(*tasks)

        save_state()

        # Broadcast round results
        await broadcast({
            "type": "round_complete",
            "round": round_num,
            "total_rounds": n_rounds,
            "teams": get_teams_summary(),
        })

        # Pause between rounds so players can follow what's happening
        await asyncio.sleep(3)

    # Game finished
    state["phase"] = "finished"
    save_state()
    summaries = get_teams_summary()
    await broadcast({
        "type": "game_finished",
        "teams": summaries,
        "bullwhip": compute_bullwhip(team_states),
        "prizes": compute_prizes(summaries),
    })


async def process_team_round(team: dict, ts: dict, round_num: int, customer_demand: int, settings: dict):
    """Process one round for one team. All 4 roles in sequence (downstream first)."""
    lead_time = settings["lead_time"]

    # Determine demand each stage receives
    # Retailer gets customer_demand; others get the ORDER placed by downstream last round
    demands = {}
    for i, role in enumerate(ROLES):
        if role == "Retailer":
            demands[role] = customer_demand
        else:
            # demand from downstream = downstream's last order
            downstream = ROLES[i - 1]  # Retailer < Wholesaler < Distributor < Manufacturer
            prev_rounds = state["rounds_data"][team["id"]][downstream]
            if prev_rounds:
                demands[role] = prev_rounds[-1]["order"]
            else:
                demands[role] = settings.get("step_demand_before", 4)

    # Process each role and collect AI calls
    ai_tasks = []
    for role in ROLES:
        rs = ts[role]

        # 1. Receive incoming shipment (front of pipeline)
        incoming = rs["pipeline"].pop(0) if rs["pipeline"] else 0
        rs["inventory"] += incoming

        # 2. Process demand: fulfill from inventory, remainder becomes backlog
        demand = demands[role]
        rs["demands_received"].append(demand)
        total_demand = demand + rs["backlog"]
        shipped = min(rs["inventory"], total_demand)
        rs["inventory"] -= shipped
        rs["backlog"] = total_demand - shipped

        # Build round data for AI
        rd = {
            "round": round_num,
            "demand": demand,
            "inventory": rs["inventory"],
            "backlog": rs["backlog"],
            "pipeline": sum(rs["pipeline"]),
            "outstanding": rs["outstanding"],
            "incoming_shipment": incoming,
            "cumulative_cost": rs["cumulative_cost"],
        }

        # Get player's strategy for this role
        player_id = team["role_map"][role]
        player = state["players"][player_id]
        strat = player["strategies"].get(role, "rational")
        custom_prompt = player.get("custom_prompts", {}).get(role, "")

        history = state["rounds_data"][team["id"]][role]

        if strat == "base_stock":
            order = base_stock_order(history, lead_time)
            reasoning = f"[Base-Stock formula] Calculated order: {order}"
            async def _bs(o=order, r=reasoning): return (o, r)
            ai_tasks.append((role, _bs()))
        else:
            # Build prompts
            if strat == "custom" and custom_prompt:
                objective = f"Objective (custom instructions from student): {custom_prompt}"
            else:
                objective = OBJECTIVES.get(strat, OBJECTIVES["rational"])

            sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                rounds=settings["rounds"],
                lead_time=lead_time,
                holding_cost=settings["holding_cost"],
                backlog_cost=settings["backlog_cost"],
                objective=objective,
            )
            user_msg = build_user_message(role, round_num, rd, history)
            ai_tasks.append((role, call_claude(state["api_key"], sys_prompt, user_msg)))

    # Run all AI calls concurrently
    results = {}
    coros = [(role, coro) for role, coro in ai_tasks]
    gathered = await asyncio.gather(*[c for _, c in coros])
    for (role, _), (order, reasoning) in zip(coros, gathered):
        results[role] = (order, reasoning)

    # Apply orders and costs
    for role in ROLES:
        rs = ts[role]
        order, reasoning = results[role]
        demand = demands[role]

        # Record the order
        rs["orders_placed"].append(order)

        # Add order to pipeline of upstream supplier
        # For Manufacturer, supplier always ships full order after lead_time
        if role == "Manufacturer":
            rs["pipeline"].append(order)
        else:
            # upstream is the next role in the chain
            upstream_idx = ROLES.index(role) + 1
            upstream_role = ROLES[upstream_idx]
            # The upstream will process this next round as demand
            # For now, assume upstream ships what it can — handled by pipeline
            rs["pipeline"].append(order)  # simplified: orders arrive after lead_time

        # Costs for this round
        round_cost = settings["holding_cost"] * rs["inventory"] + settings["backlog_cost"] * rs["backlog"]
        rs["cumulative_cost"] += round_cost

        # Save round data
        incoming = 0
        prev = state["rounds_data"][team["id"]][role]
        state["rounds_data"][team["id"]][role].append({
            "round": round_num,
            "demand": demand,
            "inventory": rs["inventory"],
            "backlog": rs["backlog"],
            "pipeline": sum(rs["pipeline"]),
            "outstanding": rs["outstanding"],
            "order": order,
            "cost": round_cost,
            "cumulative_cost": rs["cumulative_cost"],
            "reasoning": reasoning,
            "incoming_shipment": incoming,
        })


def compute_bullwhip(team_states: dict) -> dict:
    """Compute bullwhip ratio per team per role."""
    result = {}
    for team in state["teams"]:
        ts = team_states[team["id"]]
        team_bw = {}
        for role in ROLES:
            rs = ts[role]
            orders = rs["orders_placed"]
            demands = rs["demands_received"]
            if len(demands) > 1 and len(orders) > 1:
                var_orders = sum((o - sum(orders)/len(orders))**2 for o in orders) / len(orders)
                var_demands = sum((d - sum(demands)/len(demands))**2 for d in demands) / len(demands)
                team_bw[role] = round(var_orders / var_demands, 2) if var_demands > 0 else 0.0
            else:
                team_bw[role] = 0.0
        result[team["id"]] = team_bw
    return result


def get_teams_summary() -> list[dict]:
    """Build a summary of all teams. Scoring excludes warmup rounds."""
    warmup = int(state["settings"].get("warmup_rounds", 0) or 0)
    summaries = []
    for team in state["teams"]:
        roles_info = []
        total_cost = 0.0          # raw cumulative (all rounds)
        scored_total = 0.0        # excludes warmup rounds
        for role in ROLES:
            rd_list = state["rounds_data"].get(team["id"], {}).get(role, [])
            cost = rd_list[-1]["cumulative_cost"] if rd_list else 0.0
            # scored cost = sum of round costs where round > warmup
            scored_cost = sum(r.get("cost", 0.0) for r in rd_list if r.get("round", 0) > warmup)
            total_cost += cost
            scored_total += scored_cost
            player_id = team["role_map"][role]
            player = state["players"][player_id]
            strat = player["strategies"].get(role, "rational")
            roles_info.append({
                "role": role,
                "player_id": player_id,
                "player_name": player["name"],
                "strategy": strat,
                "cost": round(cost, 2),
                "scored_cost": round(scored_cost, 2),
                "rounds": rd_list,
            })
        summaries.append({
            "id": team["id"],
            "name": team["name"],
            "members": team["members"],
            "total_cost": round(total_cost, 2),
            "scored_cost": round(scored_total, 2),
            "warmup_rounds": warmup,
            "roles": roles_info,
        })
    # Sort by scored cost (the cost that actually counts)
    summaries.sort(key=lambda t: t["scored_cost"])
    return summaries


def compute_prizes(summaries: list[dict]) -> dict:
    """Compute winners: best team and best individual player per role."""
    prizes = {"best_team": None, "best_per_role": {}}
    if not summaries:
        return prizes
    best = summaries[0]
    prizes["best_team"] = {
        "team_id": best["id"],
        "team_name": best["name"],
        "scored_cost": best["scored_cost"],
        "player_names": [state["players"][pid]["name"] for pid in best["members"] if pid in state["players"]],
    }
    for role in ROLES:
        best_entry = None
        for team in summaries:
            for r in team["roles"]:
                if r["role"] != role:
                    continue
                if best_entry is None or r["scored_cost"] < best_entry["scored_cost"]:
                    best_entry = {
                        "role": role,
                        "player_id": r["player_id"],
                        "player_name": r["player_name"],
                        "team_name": team["name"],
                        "scored_cost": r["scored_cost"],
                    }
        if best_entry:
            prizes["best_per_role"][role] = best_entry
    return prizes


# ── WebSocket broadcasting ────────────────────────────────────────────────────

async def broadcast(msg: dict):
    dead = []
    text = json.dumps(msg)
    for ws in ws_clients:
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_clients.remove(ws)


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_state()
    yield
    save_state()

app = FastAPI(lifespan=lifespan)

# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/session")
async def create_session(body: dict):
    api_key = body.get("api_key", "").strip()
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return JSONResponse({"error": "API key required"}, status_code=400)

    # Guard: refuse to clobber an active session unless explicitly forced.
    force = bool(body.get("force"))
    if not force and state.get("session_id") and state.get("phase") not in (None, "setup", "finished"):
        return JSONResponse(
            {
                "error": "active_session",
                "message": "An active game session already exists.",
                "phase": state.get("phase"),
                "player_count": len(state.get("players", {})),
                "created_at": state.get("created_at"),
            },
            status_code=409,
        )

    # Apply settings if provided
    settings_update = body.get("settings", {})
    for k, v in settings_update.items():
        if k in state["settings"]:
            state["settings"][k] = v

    state["api_key"] = api_key
    state["session_id"] = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    state["phase"] = "lobby"
    state["players"] = {}
    state["teams"] = []
    state["rounds_data"] = {}
    state["current_round"] = 0
    state["created_at"] = time.time()
    state["demand_sequence"] = generate_demand_sequence(state["settings"])

    save_state()
    await broadcast({"type": "session_created", "session_id": state["session_id"]})
    return {"session_id": state["session_id"], "settings": state["settings"]}


@app.post("/api/settings")
async def update_settings(body: dict):
    """Professor updates game settings (only in lobby phase)."""
    if state["phase"] not in ("lobby", "setup"):
        return JSONResponse({"error": "Can only change settings before starting"}, status_code=400)
    for k, v in body.items():
        if k in state["settings"]:
            state["settings"][k] = v
    state["demand_sequence"] = generate_demand_sequence(state["settings"])
    save_state()
    await broadcast({"type": "settings_updated", "settings": state["settings"]})
    return {"settings": state["settings"]}


@app.post("/api/join")
async def join_game(body: dict):
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "Name required"}, status_code=400)
    if state["phase"] != "lobby":
        return JSONResponse({"error": "Game not in lobby phase"}, status_code=400)

    # If a player with this exact name already exists, return their existing id
    # so that a rejoin (from localStorage-less new device) still works.
    for pid, p in state["players"].items():
        if p["name"].lower() == name.lower():
            return {"player_id": pid, "name": p["name"], "rejoined": True}

    player_id = str(uuid.uuid4())[:8]
    state["players"][player_id] = {
        "name": name,
        "roles": [],
        "strategies": {},
        "custom_prompts": {},
        "locked_in": False,
        "ready_to_start": False,
        "games_played": 0,
    }
    save_state()
    await broadcast({
        "type": "player_joined",
        "player_id": player_id,
        "name": name,
        "player_count": len(state["players"]),
        "players": {pid: {"name": p["name"]} for pid, p in state["players"].items()},
    })
    return {"player_id": player_id, "name": name}


@app.post("/api/start")
async def start_game():
    """Professor starts — assigns teams and roles, moves to design phase."""
    if state["phase"] != "lobby":
        return JSONResponse({"error": "Not in lobby"}, status_code=400)
    if not state["players"]:
        return JSONResponse({"error": "No players"}, status_code=400)

    state["teams"] = assign_teams(state["players"])
    state["phase"] = "designing"
    save_state()

    teams_info = []
    for team in state["teams"]:
        members = []
        for pid in team["members"]:
            p = state["players"][pid]
            members.append({"id": pid, "name": p["name"], "roles": p["roles"]})
        teams_info.append({"id": team["id"], "name": team["name"], "members": members, "role_map": team["role_map"]})

    await broadcast({"type": "game_started", "teams": teams_info})
    return {"teams": teams_info}


@app.post("/api/lock-in")
async def lock_in(body: dict):
    player_id = body.get("player_id")
    if not player_id or player_id not in state["players"]:
        return JSONResponse({"error": "Invalid player"}, status_code=400)
    if state["phase"] != "designing":
        return JSONResponse({"error": "Not in design phase"}, status_code=400)

    player = state["players"][player_id]
    strategies = body.get("strategies", {})
    custom_prompts = body.get("custom_prompts", {})

    for role in player["roles"]:
        player["strategies"][role] = strategies.get(role, "rational")
        if strategies.get(role) == "custom":
            player["custom_prompts"][role] = custom_prompts.get(role, "")

    player["locked_in"] = True
    save_state()

    total = len(state["players"])
    locked = sum(1 for p in state["players"].values() if p["locked_in"])

    await broadcast({
        "type": "player_locked_in",
        "player_id": player_id,
        "locked_count": locked,
        "total_count": total,
    })

    return {"locked": locked, "total": total}


@app.post("/api/unlock")
async def unlock(body: dict):
    """Player wants to go back and edit strategies before the game starts."""
    player_id = body.get("player_id")
    if not player_id or player_id not in state["players"]:
        return JSONResponse({"error": "Invalid player"}, status_code=400)
    if state["phase"] != "designing":
        return JSONResponse({"error": "Game already running"}, status_code=400)
    player = state["players"][player_id]
    player["locked_in"] = False
    player["ready_to_start"] = False
    save_state()
    total = len(state["players"])
    locked = sum(1 for p in state["players"].values() if p["locked_in"])
    ready = sum(1 for p in state["players"].values() if p.get("ready_to_start"))
    await broadcast({
        "type": "player_unlocked",
        "player_id": player_id,
        "locked_count": locked,
        "ready_count": ready,
        "total_count": total,
    })
    return {"locked": locked, "ready": ready, "total": total}


@app.post("/api/ready")
async def player_ready(body: dict):
    """Each locked-in player presses Ready. Game begins when ALL players are ready."""
    player_id = body.get("player_id")
    if not player_id or player_id not in state["players"]:
        return JSONResponse({"error": "Invalid player"}, status_code=400)
    if state["phase"] != "designing":
        return JSONResponse({"error": "Not in design phase"}, status_code=400)

    player = state["players"][player_id]
    if not player.get("locked_in"):
        return JSONResponse({"error": "Must lock in strategies first"}, status_code=400)

    player["ready_to_start"] = True
    save_state()

    total = len(state["players"])
    ready = sum(1 for p in state["players"].values() if p.get("ready_to_start"))

    await broadcast({
        "type": "player_ready",
        "player_id": player_id,
        "ready_count": ready,
        "total_count": total,
    })

    # Begin when every player has pressed Ready.
    if ready >= total and total > 0:
        await _begin_game_internal()

    return {"ready": ready, "total": total}


async def _begin_game_internal():
    """Core begin logic, callable from /api/begin or auto from lock-in."""
    if state["phase"] != "designing":
        return
    state["phase"] = "playing"
    state["demand_sequence"] = generate_demand_sequence(state["settings"])

    # Count this as a game-played for every participating student
    for pid in state["players"]:
        state["players"][pid]["games_played"] = state["players"][pid].get("games_played", 0) + 1
    save_state()

    await broadcast({"type": "game_begin", "total_rounds": state["settings"]["rounds"]})
    asyncio.create_task(run_game())


@app.post("/api/begin")
async def begin_game():
    """Professor begins the game after all lock-ins."""
    if state["phase"] != "designing":
        return JSONResponse({"error": "Not in design phase"}, status_code=400)
    await _begin_game_internal()
    return {"status": "running"}


@app.post("/api/restart")
async def restart_game():
    """Same teams, go back to agent design."""
    state["phase"] = "designing"
    state["current_round"] = 0
    state["rounds_data"] = {}
    for p in state["players"].values():
        p["locked_in"] = False
        p["ready_to_start"] = False
        p["strategies"] = {}
        p["custom_prompts"] = {}

    teams_info = []
    for team in state["teams"]:
        members = []
        for pid in team["members"]:
            p = state["players"][pid]
            members.append({"id": pid, "name": p["name"], "roles": p["roles"]})
        teams_info.append({"id": team["id"], "name": team["name"], "members": members, "role_map": team["role_map"]})

    save_state()
    await broadcast({"type": "game_restarted", "teams": teams_info})
    return {"status": "designing"}


@app.post("/api/new-game")
async def new_game():
    """Full reset to lobby."""
    state["phase"] = "lobby"
    state["teams"] = []
    state["rounds_data"] = {}
    state["current_round"] = 0
    for p in state["players"].values():
        p["roles"] = []
        p["strategies"] = {}
        p["custom_prompts"] = {}
        p["locked_in"] = False
        p["ready_to_start"] = False
    save_state()
    await broadcast({"type": "new_game", "player_count": len(state["players"])})
    return {"status": "lobby"}


@app.post("/api/kick-all")
async def kick_all_players():
    """Professor action: remove ALL players, teams, and game data. Reset to empty lobby."""
    kicked_count = len(state["players"])
    state["players"] = {}
    state["teams"] = []
    state["rounds_data"] = {}
    state["current_round"] = 0
    state["phase"] = "lobby"
    save_state()
    await broadcast({"type": "kicked_all"})
    return {"status": "lobby", "kicked": kicked_count}


def _team_id_for_player(player_id: str) -> Optional[str]:
    if not player_id:
        return None
    for team in state.get("teams", []):
        if player_id in team.get("members", []):
            return team["id"]
    return None


@app.get("/api/state")
async def get_state(player_id: Optional[str] = None):
    max_games = state["settings"].get("max_games_per_player", 3)
    players_info = {pid: {
        "name": p["name"],
        "roles": p["roles"],
        "locked_in": p["locked_in"],
        "ready_to_start": p.get("ready_to_start", False),
        "strategies": p["strategies"],
        "games_played": p.get("games_played", 0),
        "games_remaining": max(0, max_games - p.get("games_played", 0)),
    } for pid, p in state["players"].items()}
    teams_info = []
    for team in state["teams"]:
        members = []
        for pid in team["members"]:
            p = state["players"][pid]
            members.append({"id": pid, "name": p["name"], "roles": p["roles"]})
        teams_info.append({"id": team["id"], "name": team["name"], "members": members, "role_map": team["role_map"]})

    summaries = get_teams_summary() if state["rounds_data"] else []

    # Privacy: students only get rounds_data for THEIR own team. Other teams'
    # per-round detail (orders, reasoning, inventory) stays server-side. The
    # professor (no player_id) gets everything.
    own_team_id = _team_id_for_player(player_id) if player_id else None
    if player_id:
        rounds_data_out = {own_team_id: state["rounds_data"].get(own_team_id, {})} if own_team_id else {}
    else:
        rounds_data_out = state["rounds_data"]

    return {
        "phase": state["phase"],
        "session_id": state["session_id"],
        "settings": state["settings"],
        "players": players_info,
        "teams": teams_info,
        "current_round": state["current_round"],
        "total_rounds": state["settings"]["rounds"],
        "rounds_data": rounds_data_out,
        "teams_summary": summaries,
        "prizes": compute_prizes(summaries) if state["phase"] == "finished" else None,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ws_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        if ws in ws_clients:
            ws_clients.remove(ws)


# ── Main ──────────────────────────────────────────────────────────────────────

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


if __name__ == "__main__":
    ip = get_local_ip()
    port = int(os.environ.get("PORT", 8000))
    print(f"\n{'='*60}")
    print(f"  Beer Game Classroom App")
    print(f"  Professor:  http://localhost:{port}")
    print(f"  Students:   http://{ip}:{port}/?join=true")
    print(f"{'='*60}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
