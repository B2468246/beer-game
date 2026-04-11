#!/bin/bash
# Startet den Beer-Game-Server lokal auf http://localhost:8000
# Doppelklick im Finder, um zu starten. Terminal schliessen zum Beenden.
# Wechselt ins Verzeichnis dieses Scripts, damit der Launcher
# portabel ist (funktioniert unabhängig vom Speicherort).
cd "$(dirname "$0")" || exit 1
echo ""
echo "=============================================="
echo "  Beer Game Server startet..."
echo "=============================================="
echo ""
echo "  Professor:  http://localhost:8000"
echo "  Studenten:  http://localhost:8000/?join=true"
echo ""
echo "  Zum Beenden: Strg+C druecken oder Fenster schliessen"
echo "=============================================="
echo ""
exec python3 server.py
