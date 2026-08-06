# MT5 Demo Setup -- ZeroMQ Bridge (native macOS/Linux + MT5)

This is the path that actually works from this Mac: the strategy logic
(`src/`) runs natively here, and talks to a small MQL5 Expert Advisor
inside a real MT5 terminal over a local ZeroMQ socket. **Two independent
setup tracks** -- do the Python side first (fully testable right now,
no MT5 needed), then the MT5/EA side (needs an actual MT5 terminal
somewhere).

There's also an older, Windows-native path (`src/mt5_executor.py` /
`src/mt5_live.py` originally used the `MetaTrader5` Python package
directly) that's still in the repo, unused by default, as a reference in
case anyone ever runs this on native Windows instead. This document
covers the ZeroMQ path, which is the one `mt5_live.py` actually uses now.

## Architecture

```
src/mt5_live.py  --(function calls)-->  src/mt5_zmq_bridge.py
                                              |
                                    ZeroMQ REQ/REP socket
                                     (tcp://host:5555, JSON)
                                              |
                              mt5_bridge_ea/AmaroZmqBridge.mq5
                                   (running inside MT5 terminal)
                                              |
                                     MT5's own trade API
                                              |
                                     your broker's demo account
```

Only the bottom two layers need an MT5 runtime. Everything above the
socket -- the crossover strategy, the memory system, position sizing,
ATR stops, decision logging -- is ordinary Python running right here.

**Genuine side benefit of this architecture: no MT5 login credentials
ever pass through Python, a config file, or this chat.** You log into
the MT5 terminal once, manually, via its own GUI. The EA then operates
against that already-authenticated session -- there's no `.env`,
`MT5_LOGIN`, or `MT5_PASSWORD` anywhere in this path.

## Part 1: Python side (do this now, on this Mac)

Already done as of this setup:
- `pyzmq` is in `requirements.txt` and installed.
- `src/mt5_zmq_bridge.py` implements the socket client (connect, order
  placement, position queries, rate/tick fetches) and has been verified
  end-to-end against a dummy responder standing in for the EA -- every
  command, plus the timeout-and-reconnect path, passed.
- `src/mt5_live.py` runs the actual strategy+memory decision loop
  against the bridge.

Nothing further is needed here until Part 2 gives you something to
connect to.

## Part 2: MT5 + EA side (needs an MT5 terminal)

MetaQuotes doesn't ship a native macOS MT5 terminal. Your options, in
order of how commonly brokers support them:

1. **Your broker's own "MT5 for Mac" build**, if they offer one (many
   brokers provide an installer that behaves like a native Mac app --
   it's a Wine wrapper under the hood, but you don't manage Wine
   yourself). Check your broker's download page.
2. **A Windows VM** (Parallels/VMware/UTM) on this Mac.
3. **A Windows VPS**, if you want it running independent of this
   machine being on.

Whichever you pick, the steps from there are the same:

1. **Install the MT5 terminal** and log into your **demo** account
   manually via its own login screen. Confirm you can see live prices.

2. **Install the `mql-zmq` library** (the MQL5 ZeroMQ binding the EA
   depends on -- not vendored in this repo; search for "mql-zmq" or
   "MQL5 ZeroMQ library" and get it from its own source). Follow that
   project's own install instructions -- typically: its `Zmq.mqh` (and
   supporting includes) go in `MQL5/Include/Zmq/`, and `libzmq.dll`
   (plus `libsodium.dll` if the version you get needs it) go in
   `MQL5/Libraries/`, inside the terminal's data folder (File -> Open
   Data Folder in MT5).

3. **Copy `mt5_bridge_ea/AmaroZmqBridge.mq5`** into
   `MQL5/Experts/` in that same data folder.

4. **Enable DLL imports** -- Tools -> Options -> Expert Advisors ->
   check "Allow DLL imports" (unchecked by default; the ZMQ library is a
   native DLL, so the EA won't be able to load it without this).

5. **Compile** `AmaroZmqBridge.mq5` in MetaEditor (F7). This is the step
   most likely to need small fixes -- the EA's own trading logic (order
   placement, position queries, account checks) is written against
   MQL5's standard, stable, MetaQuotes-documented API, but the
   ZMQ-library calls (`bind`/`recv`/`send`) target that library's
   commonly published usage pattern, which can shift slightly between
   versions. If it doesn't compile cleanly, the fix is almost certainly
   in those few lines, not the trading logic.

6. **Attach it to any one chart** (symbol/timeframe don't matter -- the
   EA polls on its own timer, not on price ticks). Confirm **AutoTrading**
   is enabled (top toolbar) and the EA shows a smiley face in the chart's
   top-right corner, not a frown.

7. **Check the Experts log tab** for:
   ```
   AmaroZmqBridge: bound tcp://*:5555 on DEMO account <login> (<server>), balance ...
   ```
   If instead you see "REFUSING TO INITIALIZE -- account ... is NOT a
   demo account", the EA has correctly detected a non-demo account and
   refused to load -- stop and fix the account, don't try to work around
   it.

## Part 3: connect and smoke-test

From this Mac (or wherever the Python side runs), once the EA's log
shows it's bound and waiting:

```
python -m src.mt5_zmq_bridge --host <VM-or-terminal-IP> --port 5555
```

Use `--host 127.0.0.1` if the terminal is running on this same machine;
use the VM's actual IP (check its network settings) if it's in a VM --
most VM software gives the VM an IP reachable from the host over a
NAT/bridged/host-only adapter. Expected output: connects, prints your
demo account's login/server/balance, "Trade mode: 0 (0 = demo)",
disconnects, "No orders were placed." Fix connectivity here (firewall,
IP, port) before moving on -- if this doesn't work, nothing downstream
will either.

## Part 4: run the live driver

Bounded smoke test first:
```
python -m src.mt5_live --symbol XAUUSD --mt5-symbol XAUUSD --notional 100 \
    --risk-pct 1.0 --host <VM-or-terminal-IP> --max-iterations 1
```
`--mt5-symbol` matters if your broker names gold something other than
`XAUUSD` (e.g. `XAUUSD.a`, `GOLD`) -- check Market Watch in the terminal
for the exact name. Expect a SKIP most of the time -- this is a daily
crossover strategy that trades roughly monthly, not every poll. Once
that runs cleanly, drop `--max-iterations` to let it run continuously
(default `--poll-seconds 3600` -- checks for a new closed daily candle
once an hour):
```
python -m src.mt5_live --symbol XAUUSD --mt5-symbol XAUUSD --notional 100 \
    --risk-pct 1.0 --host <VM-or-terminal-IP>
```

Check results in two places:
- `data/mt5_live_trades.csv` -- every decision this script makes
  (SKIP/BUY/SELL), including the ATR-based stop/target and lot size.
- The MT5 terminal's own account history / Trade tab -- the
  authoritative record, since real fills, requotes, and swap can differ
  slightly from what the script estimates.

## Why this is safe (no real money is at risk)

- **The EA refuses to initialize at all** on a non-demo account (see
  `OnInit` in `AmaroZmqBridge.mq5`) -- a real-money account can never
  even load this EA, independent of anything the Python side does.
- `mt5_zmq_bridge.connect()` also checks the account info it receives
  and refuses to proceed if it's somehow not a demo account, as defense
  in depth. Neither this nor the EA-side guard is bypassable from the
  CLI -- there is no `--allow-live` flag anywhere in this path.
- Every order carries an ATR-based stop-loss and take-profit (the
  backtested strategy itself has no stop -- see the "no explicit
  stop-loss" caveat in `data/performance_report.md`).
- Position sizing is risk-based (`--risk-pct`, default 1% of account
  balance per trade), not "buy as much as the notional allows."
- **The ZeroMQ socket itself has no authentication or encryption.**
  Only bind/connect it on `127.0.0.1` or a private/host-only VM network
  -- never expose port 5555 on a public interface. Anyone who can reach
  it can place orders on the connected account.

## Before trusting any of this with real capital later

Re-read `data/performance_report.md` first. As of the last full
evaluation, **no strategy in this project has a demonstrated,
statistically significant edge over simply buying and holding the
asset** -- the gold "memory mode" result is the closest thing to a real
signal and it still underperforms buy-and-hold over the backtested
period. Demo/paper trading here is meant to build genuine forward-test
evidence on data the strategy has never seen, not to skip the step of
first proving there's an edge worth trading.
