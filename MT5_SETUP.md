# MT5 Demo Setup -- Windows-Only

This is the exact, step-by-step path to get the trading robot placing real
(demo-account, paper-money) orders on MT5. Everything up to this point in
the project (backtests, Monte Carlo, alpha testing, the Deriv paper-feed
forward-test) runs fine on Mac/Linux. **This part does not** -- the
`MetaTrader5` Python package only ships for Windows, confirmed by direct
`pip install` failure on this Mac. There is no workaround; it has to run
on a Windows machine or Windows VM.

## What you'll end up running

Two scripts, in order:
1. `python -m src.mt5_executor` -- a connectivity smoke test. Connects,
   prints account info, disconnects. **Places no orders.** Run this first
   and don't move on until it succeeds.
2. `python -m src.mt5_live --symbol XAUUSD --notional 100 --risk-pct 1.0`
   -- the actual live driver. Runs the same crossover + memory decision
   logic used everywhere else in this project, and places real orders on
   your MT5 **demo** account when a signal fires. It refuses to run
   against anything but a demo account (see "Why this is safe" below).

## Step by step

1. **Get a Windows environment.** A real Windows PC, a Windows VM
   (Parallels/VMware/VirtualBox on this Mac), or a cheap Windows VPS all
   work. It needs to stay running continuously for the bot to actually
   monitor daily candles.

2. **Install an MT5 terminal** from your broker (the one whose demo
   account you're using) and log into it once manually with your demo
   credentials, just to confirm the demo account itself works and you can
   see live prices. Note the exact broker **server name** shown on the
   login screen (e.g. `HFMarketsGlobal-Demo`) -- you'll need it exactly.

3. **Install Python 3.10+ on Windows** (from python.org -- check "Add
   Python to PATH" during install).

4. **Get this repo onto the Windows machine** (git clone, or copy the
   folder) and install dependencies:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
   `requirements.txt` already conditions `MetaTrader5` on
   `sys_platform == "win32"`, so this will actually install it here
   (it's silently skipped on Mac/Linux).

5. **Create your `.env` file** (copy `.env.example` to `.env`) and fill in
   your real demo account's login, password, and server name:
   ```
   copy .env.example .env
   notepad .env
   ```
   **Do this by typing directly into the file on that machine. Never
   paste real credentials into a chat with me or anyone else, and never
   commit `.env`** -- it's already gitignored.

6. **Run the connectivity smoke test:**
   ```
   python -m src.mt5_executor
   ```
   Expected output: connects, prints your demo account's login/server/
   balance, "Trade mode: 0 (0 = demo)", disconnects, "No orders were
   placed." If this fails, fix it here before going any further --
   common causes are a wrong server name (must match the terminal's login
   screen exactly) or the MT5 terminal app not being installed/running.

7. **Run the live driver**, starting with a bounded smoke test before
   letting it run unattended:
   ```
   python -m src.mt5_live --symbol XAUUSD --mt5-symbol XAUUSD --notional 100 --risk-pct 1.0 --max-iterations 1
   ```
   `--mt5-symbol` matters if your broker names gold something other than
   `XAUUSD` (e.g. `XAUUSD.a`, `GOLD`) -- check Market Watch in the
   terminal for the exact name. Once that single poll runs cleanly (it
   will report a SKIP most of the time -- this strategy trades roughly
   monthly, not daily), drop `--max-iterations` to let it run
   continuously:
   ```
   python -m src.mt5_live --symbol XAUUSD --mt5-symbol XAUUSD --notional 100 --risk-pct 1.0
   ```
   Leave it running (e.g. in a scheduled task or just a terminal window
   that stays open) -- it polls hourly by default (`--poll-seconds 3600`)
   for a newly-closed daily candle.

8. **Check results in two places:**
   - `data/mt5_live_trades.csv` -- every decision this script makes
     (SKIP/BUY/SELL), including the ATR-based stop/target and lot size.
   - Your MT5 terminal's own account history / Trade tab -- the
     authoritative record of what actually happened, since real order
     fills, requotes, and swap can differ slightly from what the script
     estimates.

## Why this is safe (no real money is at risk)

- `mt5_executor.connect()` reads the account's `trade_mode` from MT5
  itself and **refuses to proceed** unless it's flagged as a demo
  account. This isn't a flag you set -- there's no `--allow-live` option
  exposed anywhere in `mt5_live.py`'s CLI at all. To ever point this at a
  real-money account would require someone deliberately editing
  `mt5_executor.connect()`'s call site in the code -- not something that
  can happen by accident or by running these commands as documented.
- Every order carries an ATR-based stop-loss and take-profit (the
  backtested strategy itself has no stop -- see the "no explicit
  stop-loss" caveat in `data/performance_report.md` -- this adds one
  specifically because real orders shouldn't go out unprotected, demo or
  not).
- Position sizing is risk-based (`--risk-pct`, default 1% of account
  balance per trade), not "buy as much as the notional allows" -- so a
  wrong stop distance can't blow up the account.

## Before trusting any of this with real capital later

Re-read `data/performance_report.md` first. As of the last full
evaluation, **no strategy in this project has a demonstrated,
statistically significant edge over simply buying and holding the
asset** -- the gold "memory mode" result is the closest thing to a real
signal and it still underperforms buy-and-hold over the backtested
period. Demo/paper trading here is meant to build genuine forward-test
evidence on data the strategy has never seen, not to skip the step of
first proving there's an edge worth trading.
