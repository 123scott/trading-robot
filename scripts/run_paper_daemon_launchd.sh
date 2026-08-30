#!/usr/bin/env bash
#
# run_paper_daemon_launchd.sh -- launchd entry point for the paper-trading
# daemon. Not meant to be run manually (use nohup directly for that, or see
# MT5_SETUP.md / the LaunchAgent plist in this directory for the
# supervised/keep-alive setup this script is part of).
#
# Two things this adds on top of just invoking python3 directly:
#   1. `caffeinate -i` wraps the actual daemon process, holding a
#      prevent-idle-sleep assertion for exactly as long as the daemon runs.
#      This is the missing piece plain `nohup` never provided -- nohup only
#      protects against the controlling terminal closing, it does nothing
#      about the SYSTEM going to sleep, which silently pauses (and on some
#      sleep/wake cycles, effectively kills) any background process,
#      including a `nohup`'d one. `-i` (not `-s`) specifically because `-s`
#      only asserts on AC power -- `-i` prevents idle sleep on battery too,
#      matching "runs whenever connected to Wi-Fi" regardless of power state.
#      Deliberately NOT `-d` (display sleep) -- there's no reason to force
#      the screen to stay on for a headless daemon.
#   2. Absolute paths to this repo and its venv's python3, since launchd
#      does not run this through a login shell -- PATH, cwd, and any
#      venv activation you'd normally rely on interactively are not present.
#
# Auto-restart-on-crash and start-at-login are handled by the LaunchAgent
# plist that invokes this script (KeepAlive + RunAtLoad), not by this
# script itself.

set -euo pipefail

REPO_ROOT="/Users/admin/Desktop - admin’s MacBook Pro/AMARO"
PYTHON_BIN="$REPO_ROOT/venv/bin/python3"

cd "$REPO_ROOT"
exec caffeinate -i "$PYTHON_BIN" -u scripts/run_paper_daemon.py "$@"
