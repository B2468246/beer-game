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
from contextvars import ContextVar
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Multi-session game state (in-memory + per-session JSON snapshots) ────────
#
# We store one state dict per session inside `sessions`. A ContextVar binds the
# "currently active" session for each request/task, and a thin StateProxy
# routes legacy `state["x"]` access through that ContextVar so the existing
# game-engine functions don't need to be rewritten.

ROLES = ["Retailer", "Wholesaler", "Distributor", "Manufacturer"]


def default_settings() -> dict:
    return {
        "rounds": 10,
        "lead_time": 2,
        "initial_inventory": 12,
        "initial_pipeline": 4,
        "holding_cost": 0.50,
        "backlog_cost": 1.00,
        "demand_type": "step",       # "step" or "step_variance"
        "step_demand_before": 4,
        "step_demand_after": 8,
        "step_round": 5,
        "demand_std": 2,
        "total_games": 3,
        "play_mode": "ai",           # "ai" or "human"
    }


def make_empty_session(session_id: str, name: str = "") -> dict:
    return {
        "session_id": session_id,
        "name": name or f"Session {session_id}",
        "phase": "lobby",  # lobby | designing | playing | finished
        "settings": default_settings(),
        "players": {},
        "teams": [],
        "rounds_data": {},
        "game_history": [],  # list of completed game snapshots
        "current_round": 0,
        "demand_sequence": [],
        "created_at": time.time(),
        "updated_at": time.time(),
    }


sessions: dict[str, dict] = {}                      # session_id -> state dict
session_ws_clients: dict[str, list[WebSocket]] = {} # session_id -> [ws, ...]

# Human-mode order collection per session.
# Structure: {session_id: {team_id: {role: order_value}}}
pending_human_orders: dict[str, dict] = {}
# Event fired when all orders for current round are in.
human_orders_ready: dict[str, asyncio.Event] = {}

# Single API key shared across sessions for the duration of the container.
# Kept in memory only; never persisted to disk.
api_key_global: Optional[str] = os.environ.get("ANTHROPIC_API_KEY") or None

_current_session_id: ContextVar[Optional[str]] = ContextVar("current_session_id", default=None)


class StateProxy:
    """Dict-like view onto sessions[ContextVar('current_session_id')]."""

    def _bound(self) -> dict:
        sid = _current_session_id.get()
        if sid is None or sid not in sessions:
            raise KeyError("No session bound to this request/task")
        return sessions[sid]

    def __getitem__(self, key):
        return self._bound()[key]

    def __setitem__(self, key, value):
        d = self._bound()
        d[key] = value
        d["updated_at"] = time.time()

    def __contains__(self, key):
        try:
            return key in self._bound()
        except KeyError:
            return False

    def get(self, key, default=None):
        try:
            return self._bound().get(key, default)
        except KeyError:
            return default

    def setdefault(self, key, default):
        return self._bound().setdefault(key, default)


state = StateProxy()


# ── Persistence (one JSON file per session) ──────────────────────────────────
# Render free tier has ephemeral storage; survives WS drops + browser-close
# reconnects within the container lifetime, and across deploys ONLY if a
# persistent disk is mounted at SESSIONS_DIR.

SESSIONS_DIR = os.environ.get(
    "BEERGAME_SESSIONS_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sessions"),
)


def _session_file(session_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "", session_id)
    return os.path.join(SESSIONS_DIR, f"{safe}.json")


def save_state() -> None:
    """Persist the currently-bound session to disk."""
    sid = _current_session_id.get()
    if sid is None or sid not in sessions:
        return
    try:
        os.makedirs(SESSIONS_DIR, exist_ok=True)
        with open(_session_file(sid), "w") as f:
            json.dump(sessions[sid], f)
    except Exception as e:
        print(f"[persist] save failed for {sid}: {e}")


def load_all_sessions() -> None:
    """Load every saved session from disk into memory."""
    if not os.path.isdir(SESSIONS_DIR):
        return
    loaded = 0
    for fname in os.listdir(SESSIONS_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(SESSIONS_DIR, fname)) as f:
                snapshot = json.load(f)
            sid = snapshot.get("session_id") or fname.replace(".json", "")
            # If a session was crashed mid-game, drop back to designing so
            # players can resume — the round loop won't auto-restart.
            if snapshot.get("phase") == "playing":
                snapshot["phase"] = "designing"
            sessions[sid] = snapshot
            loaded += 1
        except Exception as e:
            print(f"[persist] failed to load {fname}: {e}")
    if loaded:
        print(f"[persist] restored {loaded} session(s) from {SESSIONS_DIR}")


def delete_session_file(session_id: str) -> None:
    p = _session_file(session_id)
    try:
        if os.path.exists(p):
            os.remove(p)
    except Exception as e:
        print(f"[persist] delete failed for {session_id}: {e}")


def new_session_id() -> str:
    while True:
        sid = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
        if sid not in sessions:
            return sid


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

IMPORTANT OUTPUT FORMAT:
- The VERY FIRST line of your reply MUST be exactly: ORDER: <integer>
- After that line, explain your reasoning in as much detail as you want.
- Never omit the ORDER line — it is parsed automatically."""

DEFAULT_OBJECTIVE = "Objective: Minimize your cumulative costs by balancing inventory holding costs and backlog penalty costs. Use the student's custom instructions below to guide your ordering strategy."


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
            "max_tokens": 2048,
            "temperature": 0.0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_message}],
        }
        async with httpx.AsyncClient(timeout=60) as client:
            try:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=body,
                )
                resp.raise_for_status()
                data = resp.json()
                text = data["content"][0]["text"]
                # Parse ORDER: <number> — robust fallbacks in case the model
                # formats slightly differently or output was cut off.
                order = _parse_order(text)
                return order, text
            except Exception as e:
                return 4, f"[AI Error: {e}] Defaulting to order 4."


def _parse_order(text: str) -> int:
    """Extract the order quantity from the model's reply.

    Priority:
      1. `ORDER: <n>` (case-insensitive, anywhere in the text)
      2. `order <n>` / `order of <n>` / `ordering <n>` near the start
      3. Any standalone integer on the first line
      4. Default: 4 (safe baseline demand)
    """
    if not text:
        return 4
    m = re.search(r"ORDER\s*[:=]\s*(\d+)", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Look at the first ~200 chars for any "order <n>" pattern
    head = text[:400]
    m = re.search(r"order(?:ing)?(?:\s+of)?\s+(\d+)", head, re.IGNORECASE)
    if m:
        return int(m.group(1))
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    m = re.search(r"\b(\d+)\b", first_line)
    if m:
        return int(m.group(1))
    return 4


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
            "outstanding": 0,            # units ordered from upstream but not yet shipped
            "cumulative_cost": 0.0,
            "orders_placed": [],         # for bullwhip calculation
            "demands_received": [],
            "pending_from_downstream": 0,  # unfulfilled orders from downstream role
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

    is_human = settings.get("play_mode") == "human"
    sid = _current_session_id.get()

    for round_num in range(1, n_rounds + 1):
        state["current_round"] = round_num
        customer_demand = demand_seq[round_num - 1]

        if is_human:
            # --- Human mode: ask players for orders, then wait ---
            # Compute demand each role sees BEFORE asking for orders
            round_demands = {}  # {team_id: {role: demand}}
            for team in state["teams"]:
                ts = team_states[team["id"]]
                team_demands = {}
                for i, role in enumerate(ROLES):
                    if role == "Retailer":
                        team_demands[role] = customer_demand
                    else:
                        downstream = ROLES[i - 1]
                        prev_rounds = state["rounds_data"][team["id"]][downstream]
                        if prev_rounds:
                            team_demands[role] = prev_rounds[-1]["order"]
                        else:
                            team_demands[role] = settings.get("step_demand_before", 4)
                round_demands[team["id"]] = team_demands

            # Prepare pending orders storage
            pending_human_orders[sid] = {}
            event = asyncio.Event()
            human_orders_ready[sid] = event

            # Count total orders expected
            total_expected = len(state["teams"]) * len(ROLES)

            # Build order requests for all roles and broadcast
            order_requests = []
            for team in state["teams"]:
                ts = team_states[team["id"]]
                for role in ROLES:
                    rs = ts[role]
                    incoming = rs["pipeline"][0] if rs["pipeline"] else 0
                    player_id = team["role_map"][role]
                    order_requests.append({
                        "player_id": player_id,
                        "role": role,
                        "team_id": team["id"],
                        "team_name": team["name"],
                        "demand": round_demands[team["id"]][role],
                        "inventory": rs["inventory"] + incoming,
                        "backlog": rs["backlog"],
                        "pipeline": sum(rs["pipeline"]),
                        "outstanding": rs["outstanding"],
                        "incoming_shipment": incoming,
                        "cumulative_cost": rs["cumulative_cost"],
                        "holding_cost": settings["holding_cost"],
                        "backlog_cost": settings["backlog_cost"],
                    })
            await broadcast({
                "type": "order_request",
                "round": round_num,
                "total_rounds": n_rounds,
                "requests": order_requests,
            })

            # Also broadcast round_processing so professor sees progress
            await broadcast({
                "type": "round_processing",
                "round": round_num,
                "total_rounds": n_rounds,
                "waiting_for_orders": True,
            })

            # Wait for all orders (timeout 5 minutes)
            try:
                await asyncio.wait_for(event.wait(), timeout=300)
            except asyncio.TimeoutError:
                # Fill missing orders with demand (safe default)
                if sid not in pending_human_orders:
                    pending_human_orders[sid] = {}
                for team in state["teams"]:
                    tid = team["id"]
                    if tid not in pending_human_orders[sid]:
                        pending_human_orders[sid][tid] = {}
                    for role in ROLES:
                        if role not in pending_human_orders[sid][tid]:
                            pending_human_orders[sid][tid][role] = round_demands[tid][role]

            # Process with collected orders
            collected = pending_human_orders.pop(sid, {})
            human_orders_ready.pop(sid, None)

            tasks_list = []
            for team in state["teams"]:
                tasks_list.append(process_team_round_human(
                    team, team_states[team["id"]], round_num,
                    customer_demand, settings,
                    collected.get(team["id"], {}),
                ))
            await asyncio.gather(*tasks_list)
        else:
            # --- AI mode (original behavior) ---
            # Tell clients the round is being computed
            await broadcast({
                "type": "round_processing",
                "round": round_num,
                "total_rounds": n_rounds,
            })

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
    cumulative = get_cumulative_teams_summary()
    game_num = len(state.get("game_history", [])) + 1
    await broadcast({
        "type": "game_finished",
        "teams": summaries,
        "cumulative_teams": cumulative,
        "game_num": game_num,
        "bullwhip": compute_bullwhip(team_states),
        "prizes": compute_prizes(cumulative),  # prizes based on cumulative
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

    # ── Step 1: Upstream fulfillment of pending orders from last round ──
    # Each role tries to ship what downstream ordered last round.
    # Process upstream→downstream (Manufacturer first, then Distributor, etc.)
    round_incoming = {}   # role -> incoming shipment this round
    round_shipped = {}    # role -> units shipped to downstream this round
    for role in ROLES:
        rs = ts[role]

        # Receive incoming shipment from OWN pipeline (goods arriving after lead_time)
        incoming = rs["pipeline"].pop(0) if rs["pipeline"] else 0
        rs["inventory"] += incoming
        round_incoming[role] = incoming

    # Now each role fulfills demand from downstream's orders + own backlog
    for role in ROLES:
        rs = ts[role]
        demand = demands[role]
        rs["demands_received"].append(demand)

        # Total demand = new demand (downstream's last order) + accumulated backlog
        total_demand = demand + rs["backlog"]
        shipped = min(rs["inventory"], total_demand)
        rs["inventory"] -= shipped
        rs["backlog"] = total_demand - shipped
        round_shipped[role] = shipped

        # The shipped amount goes into the DOWNSTREAM role's pipeline
        # (not our own pipeline — downstream receives what we ship)
        role_idx = ROLES.index(role)
        if role_idx > 0:
            downstream_role = ROLES[role_idx - 1]
            ds = ts[downstream_role]
            ds["pipeline"].append(shipped)
            # Reduce downstream's outstanding by what we shipped
            ds["outstanding"] = max(0, ds["outstanding"] - shipped)

    # ── Step 2: Collect AI decisions for all roles ──
    ai_tasks = []
    for role in ROLES:
        rs = ts[role]
        demand = demands[role]

        # Build round data for AI
        rd = {
            "round": round_num,
            "demand": demand,
            "inventory": rs["inventory"],
            "backlog": rs["backlog"],
            "pipeline": sum(rs["pipeline"]),
            "outstanding": rs["outstanding"],
            "incoming_shipment": round_incoming[role],
            "cumulative_cost": rs["cumulative_cost"],
        }

        # Get player's strategy for this role
        player_id = team["role_map"][role]
        player = state["players"][player_id]
        custom_prompt = player.get("custom_prompts", {}).get(role, "")

        history = state["rounds_data"][team["id"]][role]

        objective = f"Objective (custom instructions from student): {custom_prompt}" if custom_prompt else DEFAULT_OBJECTIVE

        sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            rounds=settings["rounds"],
            lead_time=lead_time,
            holding_cost=settings["holding_cost"],
            backlog_cost=settings["backlog_cost"],
            objective=objective,
        )
        user_msg = build_user_message(role, round_num, rd, history)
        ai_tasks.append((role, call_claude(api_key_global or "", sys_prompt, user_msg)))

    # Run all AI calls concurrently
    results = {}
    coros = [(role, coro) for role, coro in ai_tasks]
    gathered = await asyncio.gather(*[c for _, c in coros])
    for (role, _), (order, reasoning) in zip(coros, gathered):
        results[role] = (order, reasoning)

    # ── Step 3: Apply orders and costs ──
    for role in ROLES:
        rs = ts[role]
        order, reasoning = results[role]
        demand = demands[role]

        # Record the order
        rs["orders_placed"].append(order)

        # Manufacturer: external supplier always ships full order after lead_time
        if role == "Manufacturer":
            rs["pipeline"].append(order)
        else:
            # Order goes to upstream — track as outstanding.
            # The upstream will try to fulfill it next round.
            rs["outstanding"] += order

        # Costs for this round
        round_cost = settings["holding_cost"] * rs["inventory"] + settings["backlog_cost"] * rs["backlog"]
        rs["cumulative_cost"] += round_cost

        # Save round data
        state["rounds_data"][team["id"]][role].append({
            "round": round_num,
            "demand": demand,
            "inventory": rs["inventory"],
            "backlog": rs["backlog"],
            "pipeline": sum(rs["pipeline"]),
            "pipeline_detail": list(rs["pipeline"]),
            "shipped": round_shipped[role],
            "outstanding": rs["outstanding"],
            "order": order,
            "cost": round_cost,
            "cumulative_cost": rs["cumulative_cost"],
            "reasoning": reasoning,
            "incoming_shipment": round_incoming[role],
        })


async def process_team_round_human(team: dict, ts: dict, round_num: int,
                                    customer_demand: int, settings: dict,
                                    human_orders: dict):
    """Process one round for one team using human-submitted orders."""
    lead_time = settings["lead_time"]

    # Determine demand each stage receives
    demands = {}
    for i, role in enumerate(ROLES):
        if role == "Retailer":
            demands[role] = customer_demand
        else:
            downstream = ROLES[i - 1]
            prev_rounds = state["rounds_data"][team["id"]][downstream]
            if prev_rounds:
                demands[role] = prev_rounds[-1]["order"]
            else:
                demands[role] = settings.get("step_demand_before", 4)

    # ── Step 1: Receive incoming shipments ──
    round_incoming = {}
    for role in ROLES:
        rs = ts[role]
        incoming = rs["pipeline"].pop(0) if rs["pipeline"] else 0
        rs["inventory"] += incoming
        round_incoming[role] = incoming

    # ── Step 2: Fulfill demand (ship to downstream) ──
    round_shipped = {}
    for role in ROLES:
        rs = ts[role]
        demand = demands[role]
        rs["demands_received"].append(demand)
        total_demand = demand + rs["backlog"]
        shipped = min(rs["inventory"], total_demand)
        rs["inventory"] -= shipped
        rs["backlog"] = total_demand - shipped
        round_shipped[role] = shipped

        # Ship to downstream's pipeline
        role_idx = ROLES.index(role)
        if role_idx > 0:
            downstream_role = ROLES[role_idx - 1]
            ds = ts[downstream_role]
            ds["pipeline"].append(shipped)
            ds["outstanding"] = max(0, ds["outstanding"] - shipped)

    # ── Step 3: Apply human orders ──
    for role in ROLES:
        rs = ts[role]
        demand = demands[role]
        order = human_orders.get(role, demand)
        rs["orders_placed"].append(order)

        # Manufacturer: external supplier always ships full order
        if role == "Manufacturer":
            rs["pipeline"].append(order)
        else:
            rs["outstanding"] += order

        # Costs
        round_cost = settings["holding_cost"] * rs["inventory"] + settings["backlog_cost"] * rs["backlog"]
        rs["cumulative_cost"] += round_cost

        # Save round data
        state["rounds_data"][team["id"]][role].append({
            "round": round_num,
            "demand": demand,
            "inventory": rs["inventory"],
            "backlog": rs["backlog"],
            "pipeline": sum(rs["pipeline"]),
            "pipeline_detail": list(rs["pipeline"]),
            "shipped": round_shipped[role],
            "outstanding": rs["outstanding"],
            "order": order,
            "cost": round_cost,
            "cumulative_cost": rs["cumulative_cost"],
            "reasoning": f"[Human] Ordered {order} units",
            "incoming_shipment": round_incoming[role],
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
    """Build a summary of all teams (current game only)."""
    summaries = []
    for team in state["teams"]:
        roles_info = []
        total_cost = 0.0          # raw cumulative (all rounds)
        for role in ROLES:
            rd_list = state["rounds_data"].get(team["id"], {}).get(role, [])
            cost = rd_list[-1]["cumulative_cost"] if rd_list else 0.0
            total_cost += cost
            player_id = team["role_map"][role]
            player = state["players"][player_id]
            strat = player["strategies"].get(role, "rational")
            roles_info.append({
                "role": role,
                "player_id": player_id,
                "player_name": player["name"],
                "strategy": strat,
                "cost": round(cost, 2),
                "scored_cost": round(cost, 2),
                "rounds": rd_list,
            })
        summaries.append({
            "id": team["id"],
            "name": team["name"],
            "members": team["members"],
            "total_cost": round(total_cost, 2),
            "scored_cost": round(total_cost, 2),
            "roles": roles_info,
        })
    # Sort by total cost (lower is better)
    summaries.sort(key=lambda t: t["scored_cost"])
    return summaries


def get_cumulative_teams_summary() -> list[dict]:
    """Build a leaderboard with costs summed across ALL games (history + current)."""
    # Accumulate historical costs per team
    history = state.get("game_history", [])
    team_cumulative: dict[str, float] = {}       # team_id -> total cost
    team_role_cumulative: dict[str, dict] = {}   # team_id -> {role -> cost}
    for entry in history:
        for ts in entry.get("teams_summary", []):
            tid = ts["id"]
            team_cumulative[tid] = team_cumulative.get(tid, 0.0) + ts["total_cost"]
            if tid not in team_role_cumulative:
                team_role_cumulative[tid] = {}
            for r in ts.get("roles", []):
                team_role_cumulative[tid][r["role"]] = team_role_cumulative[tid].get(r["role"], 0.0) + r["cost"]

    # Add current game costs
    current = get_teams_summary()
    for ts in current:
        tid = ts["id"]
        team_cumulative[tid] = team_cumulative.get(tid, 0.0) + ts["total_cost"]
        if tid not in team_role_cumulative:
            team_role_cumulative[tid] = {}
        for r in ts.get("roles", []):
            team_role_cumulative[tid][r["role"]] = team_role_cumulative[tid].get(r["role"], 0.0) + r["cost"]

    # Build cumulative summaries using current team structure
    cumulative = []
    for ts in current:
        tid = ts["id"]
        cum_roles = []
        for r in ts["roles"]:
            cum_cost = round(team_role_cumulative.get(tid, {}).get(r["role"], r["cost"]), 2)
            cum_roles.append({**r, "cost": cum_cost, "scored_cost": cum_cost})
        cum_total = round(team_cumulative.get(tid, ts["total_cost"]), 2)
        cumulative.append({**ts, "total_cost": cum_total, "scored_cost": cum_total, "roles": cum_roles})
    cumulative.sort(key=lambda t: t["scored_cost"])
    return cumulative


def archive_current_game():
    """Snapshot the current game's results into game_history before resetting."""
    if not state["rounds_data"]:
        return  # nothing to archive
    if "game_history" not in state:
        state["game_history"] = []
    game_num = len(state["game_history"]) + 1
    summaries = get_teams_summary()
    # Store a lightweight snapshot (rounds_data per team + summary)
    state["game_history"].append({
        "game_num": game_num,
        "rounds_data": state["rounds_data"],
        "teams_summary": summaries,
        "total_rounds": state["settings"]["rounds"],
    })


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


# ── WebSocket broadcasting (per-session) ──────────────────────────────────────

async def broadcast(msg: dict):
    """Send to every WS client subscribed to the currently bound session."""
    sid = _current_session_id.get()
    if sid is None:
        return
    text = json.dumps(msg)
    dead = []
    for ws in session_ws_clients.get(sid, []):
        try:
            await ws.send_text(text)
        except Exception:
            dead.append(ws)
    if dead:
        clients = session_ws_clients.get(sid, [])
        for ws in dead:
            if ws in clients:
                clients.remove(ws)


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_all_sessions()
    yield

app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def bind_session_middleware(request: Request, call_next):
    """Bind ContextVar from ?session_id= query param so legacy `state["x"]`
    access inside endpoints transparently routes to the right session."""
    sid = request.query_params.get("session_id")
    token = None
    if sid and sid in sessions:
        token = _current_session_id.set(sid)
    try:
        return await call_next(request)
    finally:
        if token is not None:
            _current_session_id.reset(token)


def _bind_session_or_400(sid: Optional[str]):
    """Helper for endpoints that REQUIRE a bound session."""
    if not sid:
        return JSONResponse({"error": "session_id required"}, status_code=400)
    if sid not in sessions:
        return JSONResponse({"error": "session not found"}, status_code=404)
    return None


# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/api-key")
async def set_api_key(body: dict):
    """Professor sets the shared Anthropic API key (in-memory only)."""
    global api_key_global
    key = (body.get("api_key") or "").strip()
    if not key:
        return JSONResponse({"error": "API key required"}, status_code=400)
    api_key_global = key
    return {"ok": True}


@app.get("/api/api-key")
async def has_api_key():
    return {"set": bool(api_key_global)}


def _session_summary(snap: dict) -> dict:
    return {
        "session_id": snap.get("session_id"),
        "name": snap.get("name") or f"Session {snap.get('session_id')}",
        "phase": snap.get("phase"),
        "player_count": len(snap.get("players") or {}),
        "team_count": len(snap.get("teams") or []),
        "current_round": snap.get("current_round", 0),
        "total_rounds": (snap.get("settings") or {}).get("rounds", 0),
        "created_at": snap.get("created_at"),
        "updated_at": snap.get("updated_at"),
    }


@app.get("/api/sessions")
async def list_sessions():
    """List every session the professor can resume or inspect."""
    items = [_session_summary(s) for s in sessions.values()]
    items.sort(key=lambda s: s.get("updated_at") or 0, reverse=True)
    return {"sessions": items, "api_key_set": bool(api_key_global)}


@app.post("/api/sessions")
async def create_session_v2(body: dict):
    """Create a brand-new session. API key must already be set via /api/api-key
    (or be provided here as a convenience)."""
    global api_key_global
    api_key = (body.get("api_key") or "").strip()
    if api_key:
        api_key_global = api_key
    if not api_key_global:
        return JSONResponse({"error": "API key required"}, status_code=400)

    name = (body.get("name") or "").strip()
    sid = new_session_id()
    snap = make_empty_session(sid, name=name)

    # Apply optional settings overrides up-front
    settings_update = body.get("settings") or {}
    for k, v in settings_update.items():
        if k in snap["settings"]:
            snap["settings"][k] = v
    snap["demand_sequence"] = generate_demand_sequence(snap["settings"])

    sessions[sid] = snap
    session_ws_clients.setdefault(sid, [])

    # Bind so save_state() targets the new session.
    token = _current_session_id.set(sid)
    try:
        save_state()
    finally:
        _current_session_id.reset(token)

    return {"session_id": sid, "name": snap["name"], "settings": snap["settings"]}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        return JSONResponse({"error": "session not found"}, status_code=404)
    # Close any subscribed WS clients gracefully
    for ws in list(session_ws_clients.get(session_id, [])):
        try:
            await ws.close(code=1000)
        except Exception:
            pass
    session_ws_clients.pop(session_id, None)
    sessions.pop(session_id, None)
    delete_session_file(session_id)
    return {"ok": True}


# Legacy endpoint kept for older clients — proxies to the v2 creator and
# returns the same shape the old frontend expected.
@app.post("/api/session")
async def create_session_legacy(body: dict):
    return await create_session_v2(body)


@app.post("/api/settings")
async def update_settings(body: dict):
    """Professor updates game settings (only in lobby phase)."""
    if state["phase"] not in ("lobby", "setup"):
        return JSONResponse({"error": "Can only change settings before starting"}, status_code=400)
    valid_keys = set(default_settings().keys())
    for k, v in body.items():
        if k in state["settings"] or k in valid_keys:
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

    teams_info = []
    for team in state["teams"]:
        members = []
        for pid in team["members"]:
            p = state["players"][pid]
            members.append({"id": pid, "name": p["name"], "roles": p["roles"]})
        teams_info.append({"id": team["id"], "name": team["name"], "members": members, "role_map": team["role_map"]})

    is_human = state["settings"].get("play_mode") == "human"
    if is_human:
        # Skip designing — go straight to playing
        state["phase"] = "designing"  # briefly set so _begin_game_internal works
        save_state()
        await broadcast({"type": "game_started", "teams": teams_info, "play_mode": "human"})
        await _begin_game_internal()
    else:
        state["phase"] = "designing"
        save_state()
        await broadcast({"type": "game_started", "teams": teams_info, "play_mode": "ai"})

    return {"teams": teams_info}


@app.post("/api/lock-in")
async def lock_in(body: dict):
    player_id = body.get("player_id")
    if not player_id or player_id not in state["players"]:
        return JSONResponse({"error": "Invalid player"}, status_code=400)
    if state["phase"] != "designing":
        return JSONResponse({"error": "Not in design phase"}, status_code=400)

    player = state["players"][player_id]
    custom_prompts = body.get("custom_prompts", {})

    for role in player["roles"]:
        player["strategies"][role] = "custom"
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
    # Validate API key before starting the game (only in AI mode)
    is_human = state["settings"].get("play_mode") == "human"
    if not is_human and not api_key_global:
        await broadcast({"type": "error", "message": "no_api_key",
                         "detail": "No Anthropic API key set. The professor must enter an API key before the game can start."})
        return
    state["phase"] = "playing"
    state["demand_sequence"] = generate_demand_sequence(state["settings"])

    # Count this as a game-played for every participating student
    for pid in state["players"]:
        state["players"][pid]["games_played"] = state["players"][pid].get("games_played", 0) + 1
    save_state()

    await broadcast({"type": "game_begin", "total_rounds": state["settings"]["rounds"],
                      "play_mode": state["settings"].get("play_mode", "ai")})

    # Capture the currently bound session id and rebind it inside the
    # background task so `state[...]`, `save_state()` and `broadcast()` keep
    # targeting the right session.
    sid = _current_session_id.get()

    async def _run_game_in_session():
        token = _current_session_id.set(sid)
        try:
            await run_game()
        finally:
            _current_session_id.reset(token)

    asyncio.create_task(_run_game_in_session())


@app.post("/api/submit-order")
async def submit_order(body: dict):
    """Human player submits their order for the current round."""
    player_id = body.get("player_id")
    team_id = body.get("team_id")
    role = body.get("role")
    order = body.get("order")
    if not all([player_id, team_id, role]) or order is None:
        return JSONResponse({"error": "Missing fields"}, status_code=400)
    order = max(0, int(order))

    sid = _current_session_id.get()
    if sid not in pending_human_orders:
        return JSONResponse({"error": "No round in progress"}, status_code=400)

    # Store the order
    pending_human_orders[sid].setdefault(team_id, {})[role] = order

    # Check if all orders are in
    total_expected = len(state["teams"]) * len(ROLES)
    total_received = sum(len(roles) for roles in pending_human_orders[sid].values())

    await broadcast({
        "type": "order_submitted",
        "player_id": player_id,
        "role": role,
        "orders_received": total_received,
        "orders_expected": total_expected,
    })

    if total_received >= total_expected:
        event = human_orders_ready.get(sid)
        if event:
            event.set()

    return {"status": "ok", "orders_received": total_received, "orders_expected": total_expected}


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
    archive_current_game()
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
    archive_current_game()
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
    state["game_history"] = []
    state["current_round"] = 0
    state["phase"] = "lobby"
    save_state()
    await broadcast({"type": "kicked_all"})
    return {"status": "lobby", "kicked": kicked_count}


@app.get("/api/export-csv")
async def export_csv():
    """Export game results as CSV for professor grading."""
    import io, csv
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Team", "Role", "Player", "Round", "Demand", "Order",
                     "Inventory", "Backlog", "Pipeline", "Shipped",
                     "Round Cost", "Cumulative Cost"])
    for team in state.get("teams", []):
        tid = team["id"]
        role_map = team.get("role_map", {})
        rd = state.get("rounds_data", {}).get(tid, {})
        for role in ROLES:
            pid = role_map.get(role, "")
            player_name = state["players"].get(pid, {}).get("name", pid) if pid else ""
            for entry in rd.get(role, []):
                writer.writerow([
                    team["name"], role, player_name,
                    entry.get("round", ""),
                    entry.get("demand", ""),
                    entry.get("order", ""),
                    entry.get("inventory", ""),
                    entry.get("backlog", ""),
                    entry.get("pipeline", ""),
                    entry.get("shipped", ""),
                    f"{entry.get('cost', 0):.2f}",
                    f"{entry.get('cumulative_cost', 0):.2f}",
                ])
    from starlette.responses import Response
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=beer_game_results.csv"},
    )


def _team_id_for_player(player_id: str) -> Optional[str]:
    if not player_id:
        return None
    for team in state.get("teams", []):
        if player_id in team.get("members", []):
            return team["id"]
    return None


@app.get("/api/state")
async def get_state(player_id: Optional[str] = None):
    if _current_session_id.get() is None:
        return JSONResponse({"error": "unknown_session"}, status_code=404)
    max_games = state["settings"].get("total_games", state["settings"].get("max_games_per_player", 3))
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
    cumulative = get_cumulative_teams_summary() if state["rounds_data"] else []
    game_num = len(state.get("game_history", [])) + (1 if state["rounds_data"] else 0)

    # Privacy: students only get rounds_data for THEIR own team. Other teams'
    # per-round detail (orders, reasoning, inventory) stays server-side. The
    # professor (no player_id) gets everything.
    own_team_id = _team_id_for_player(player_id) if player_id else None
    if player_id:
        rounds_data_out = {own_team_id: state["rounds_data"].get(own_team_id, {})} if own_team_id else {}
        # Filter game_history to only include this player's team data
        history_out = []
        for entry in state.get("game_history", []):
            filtered_rd = {own_team_id: entry["rounds_data"].get(own_team_id, {})} if own_team_id else {}
            history_out.append({**entry, "rounds_data": filtered_rd})
    else:
        rounds_data_out = state["rounds_data"]
        history_out = state.get("game_history", [])

    return {
        "phase": state["phase"],
        "session_id": state["session_id"],
        "settings": state["settings"],
        "players": players_info,
        "teams": teams_info,
        "current_round": state["current_round"],
        "total_rounds": state["settings"]["rounds"],
        "rounds_data": rounds_data_out,
        "game_history": history_out,
        "game_num": game_num,
        "teams_summary": summaries,
        "cumulative_teams": cumulative,
        "prizes": compute_prizes(cumulative) if state["phase"] == "finished" else None,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    sid = ws.query_params.get("session_id")
    await ws.accept()
    if not sid or sid not in sessions:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": "unknown_session"}))
        finally:
            await ws.close(code=1008)
        return
    clients = session_ws_clients.setdefault(sid, [])
    clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if ws in clients:
            clients.remove(ws)


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
