//+------------------------------------------------------------------+
//| AmaroZmqBridge.mq5                                                |
//|                                                                    |
//| Companion EA for src/mt5_zmq_bridge.py. Attach to ONE chart (any  |
//| symbol/timeframe -- an OnTimer poll loop drives everything, not   |
//| price ticks). Listens on a local ZeroMQ REP socket for JSON       |
//| commands and executes them against this terminal's connected      |
//| account. Full protocol documented in mt5_zmq_bridge.py's docstring|
//| and in ../MT5_SETUP.md -- read MT5_SETUP.md for the full install/ |
//| compile/attach walkthrough, including the "Allow DLL imports"     |
//| terminal setting this requires.                                   |
//|                                                                    |
//| REQUIRES the third-party "mql-zmq" library (MQL5 ZeroMQ binding). |
//| Not vendored here -- find and install it yourself (search for     |
//| "mql-zmq" / MQL5 ZeroMQ library; a well-known one is by            |
//| dingmaotu), following ITS install instructions: its Zmq.mqh (and  |
//| supporting includes) go in MQL5/Include/Zmq/, and libzmq.dll      |
//| (+ libsodium.dll if required) go in MQL5/Libraries/, before this  |
//| file will compile.                                                 |
//|                                                                    |
//| *** UNCOMPILED, UNTESTED *** -- there is no MetaEditor/MT5        |
//| terminal available in this development environment. Everything    |
//| using MQL5's own bundled trade API (CTrade, AccountInfo*,         |
//| SymbolInfo*, Position*) is written with high confidence against   |
//| MetaQuotes' documented, stable API. The Context/Socket calls       |
//| (bind/recv/send) target the mql-zmq library's commonly-published  |
//| usage pattern, but third-party library APIs shift between         |
//| versions -- that's the part most likely to need small adjustments |
//| to compile against whatever version you actually install. Verify  |
//| it compiles cleanly and test thoroughly on a DEMO account before  |
//| trusting it with anything.                                        |
//|                                                                    |
//| SAFETY: refuses to initialize AT ALL on a non-demo account (see   |
//| OnInit) -- a real-money account can never even load this EA,      |
//| independent of anything the Python side does or doesn't check.    |
//+------------------------------------------------------------------+
#property strict
#property version   "1.00"
#property description "ZeroMQ bridge for the AMARO trading bot. Demo accounts only -- see OnInit."

#include <Trade/Trade.mqh>
#include <Zmq/Zmq.mqh>

input string BindAddress    = "tcp://*:5555"; // ZMQ REP bind address -- localhost/private-network only, NEVER a public interface
input int    PollIntervalMs = 100;            // OnTimer poll frequency for incoming commands
input int    DefaultMagic   = 202607;         // must match MAGIC in src/mt5_live.py

Context zmqContext("amaro-bridge");
Socket  repSocket(zmqContext, ZMQ_REP);
CTrade  trade;

//+------------------------------------------------------------------+
int OnInit()
  {
   if((ENUM_ACCOUNT_TRADE_MODE)AccountInfoInteger(ACCOUNT_TRADE_MODE) != ACCOUNT_TRADE_MODE_DEMO)
     {
      Print("AmaroZmqBridge: REFUSING TO INITIALIZE -- account ", AccountInfoInteger(ACCOUNT_LOGIN),
            " on ", AccountInfoString(ACCOUNT_SERVER), " is NOT a demo account. ",
            "This EA only ever runs on demo accounts, by design -- see the header comment.");
      return(INIT_FAILED);
     }

   repSocket.bind(BindAddress);
   trade.SetExpertMagicNumber(DefaultMagic);
   EventSetMillisecondTimer(PollIntervalMs);

   Print("AmaroZmqBridge: bound ", BindAddress, " on DEMO account ", AccountInfoInteger(ACCOUNT_LOGIN),
         " (", AccountInfoString(ACCOUNT_SERVER), "), balance ",
         DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2), " ", AccountInfoString(ACCOUNT_CURRENCY),
         ". Waiting for commands...");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EventKillTimer();
   repSocket.unbind(BindAddress);
  }

//+------------------------------------------------------------------+
//| Poll loop. Non-blocking recv (ZMQ_DONTWAIT) -- if nothing's       |
//| waiting, return immediately and try again next timer tick.        |
//+------------------------------------------------------------------+
void OnTimer()
  {
   ZmqMsg request;
   if(!repSocket.recv(request, true))
      return;

   string json  = request.getData();
   string reply = HandleRequest(json);

   ZmqMsg response(reply);
   repSocket.send(response);
  }

//+------------------------------------------------------------------+
//| Command dispatch                                                  |
//+------------------------------------------------------------------+
string HandleRequest(string json)
  {
   string cmd = ExtractString(json, "cmd");

   if(cmd == "PING")        return HandlePing();
   if(cmd == "SYMBOL_INFO")  return HandleSymbolInfo(json);
   if(cmd == "RATES")        return HandleRates(json);
   if(cmd == "TICK")         return HandleTick(json);
   if(cmd == "ORDER")        return HandleOrder(json);
   if(cmd == "POSITIONS")    return HandlePositions(json);
   if(cmd == "CLOSE")        return HandleClose(json);

   return "{\"ok\":false,\"error\":\"Unknown cmd: " + cmd + "\"}";
  }

//+------------------------------------------------------------------+
string HandlePing()
  {
   return StringFormat(
      "{\"ok\":true,\"account\":{\"login\":%I64d,\"server\":\"%s\",\"balance\":%.2f,\"currency\":\"%s\",\"trade_mode\":%d}}",
      AccountInfoInteger(ACCOUNT_LOGIN), AccountInfoString(ACCOUNT_SERVER), AccountInfoDouble(ACCOUNT_BALANCE),
      AccountInfoString(ACCOUNT_CURRENCY), (int)AccountInfoInteger(ACCOUNT_TRADE_MODE));
  }

//+------------------------------------------------------------------+
string HandleSymbolInfo(string json)
  {
   string symbol = ExtractString(json, "symbol");
   if(!SymbolSelect(symbol, true))
      return "{\"ok\":false,\"error\":\"Unknown symbol: " + symbol + " -- check Market Watch for the exact broker name\"}";

   double tickValue = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize   = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
   double volMin     = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double volMax     = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double volStep    = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

   return StringFormat(
      "{\"ok\":true,\"tick_value\":%.5f,\"tick_size\":%.5f,\"volume_min\":%.2f,\"volume_max\":%.2f,\"volume_step\":%.2f}",
      tickValue, tickSize, volMin, volMax, volStep);
  }

//+------------------------------------------------------------------+
ENUM_TIMEFRAMES ParseTimeframe(string tf)
  {
   if(tf == "M1")  return PERIOD_M1;
   if(tf == "M5")  return PERIOD_M5;
   if(tf == "M15") return PERIOD_M15;
   if(tf == "H1")  return PERIOD_H1;
   if(tf == "H4")  return PERIOD_H4;
   return PERIOD_D1;
  }

string HandleRates(string json)
  {
   string symbol   = ExtractString(json, "symbol");
   string tfStr    = ExtractString(json, "timeframe");
   int    startPos = (int)ExtractLong(json, "start_pos");
   int    count    = (int)ExtractLong(json, "count");

   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   int copied = CopyRates(symbol, ParseTimeframe(tfStr), startPos, count, rates);
   if(copied <= 0)
      return "{\"ok\":false,\"error\":\"CopyRates returned no data for " + symbol + "/" + tfStr + "\"}";

   string arr = "";
   for(int i = copied - 1; i >= 0; i--)  // oldest-first, matching src/candle.py's ascending-time convention
     {
      if(arr != "") arr += ",";
      arr += StringFormat("{\"time\":%I64d,\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f,\"tick_volume\":%d}",
                           rates[i].time, rates[i].open, rates[i].high, rates[i].low, rates[i].close,
                           (int)rates[i].tick_volume);
     }
   return "{\"ok\":true,\"rates\":[" + arr + "]}";
  }

//+------------------------------------------------------------------+
string HandleTick(string json)
  {
   string symbol = ExtractString(json, "symbol");
   MqlTick tick;
   if(!SymbolInfoTick(symbol, tick))
      return "{\"ok\":false,\"error\":\"SymbolInfoTick failed for " + symbol + "\"}";
   return StringFormat("{\"ok\":true,\"bid\":%.5f,\"ask\":%.5f,\"time\":%I64d}", tick.bid, tick.ask, tick.time);
  }

//+------------------------------------------------------------------+
string HandleOrder(string json)
  {
   string symbol    = ExtractString(json, "symbol");
   string direction = ExtractString(json, "direction");
   double volume    = ExtractDouble(json, "volume");
   double sl        = ExtractDouble(json, "sl");
   double tp        = ExtractDouble(json, "tp");
   long   magic     = ExtractLong(json, "magic");
   string comment   = ExtractString(json, "comment");
   long   deviation = ExtractLong(json, "deviation");

   trade.SetExpertMagicNumber(magic > 0 ? magic : DefaultMagic);
   trade.SetDeviationInPoints((int)(deviation > 0 ? deviation : 20));

   bool sent = (direction == "buy")
             ? trade.Buy(volume, symbol, 0.0, sl, tp, comment)
             : trade.Sell(volume, symbol, 0.0, sl, tp, comment);

   if(!sent)
      return StringFormat("{\"ok\":false,\"error\":\"order failed, retcode=%d: %s\"}",
                           trade.ResultRetcode(), trade.ResultRetcodeDescription());

   return StringFormat(
      "{\"ok\":true,\"retcode\":%d,\"ticket\":%I64u,\"price\":%.5f,\"volume\":%.2f,\"comment\":\"%s\"}",
      trade.ResultRetcode(), trade.ResultDeal(), trade.ResultPrice(), trade.ResultVolume(),
      trade.ResultRetcodeDescription());
  }

//+------------------------------------------------------------------+
string HandlePositions(string json)
  {
   string symbol      = ExtractString(json, "symbol");
   long   magicFilter = ExtractLong(json, "magic");  // 0 == no filter (this project's real magic is never 0)

   string arr = "";
   int total = PositionsTotal();
   for(int i = 0; i < total; i++)
     {
      ulong ticket = PositionGetTicket(i);  // also selects the position for the Get* calls below
      if(ticket == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != symbol) continue;
      long magic = PositionGetInteger(POSITION_MAGIC);
      if(magicFilter != 0 && magic != magicFilter) continue;

      if(arr != "") arr += ",";
      arr += StringFormat(
         "{\"ticket\":%I64u,\"symbol\":\"%s\",\"volume\":%.2f,\"price_open\":%.5f,\"sl\":%.5f,\"tp\":%.5f,\"magic\":%I64d}",
         ticket, symbol, PositionGetDouble(POSITION_VOLUME), PositionGetDouble(POSITION_PRICE_OPEN),
         PositionGetDouble(POSITION_SL), PositionGetDouble(POSITION_TP), magic);
     }
   return "{\"ok\":true,\"positions\":[" + arr + "]}";
  }

//+------------------------------------------------------------------+
string HandleClose(string json)
  {
   ulong ticket    = (ulong)ExtractLong(json, "ticket");
   long  deviation = ExtractLong(json, "deviation");

   if(!PositionSelectByTicket(ticket))
      return StringFormat("{\"ok\":false,\"error\":\"No open position with ticket %I64u\"}", ticket);

   trade.SetDeviationInPoints((int)(deviation > 0 ? deviation : 20));
   bool sent = trade.PositionClose(ticket, (int)(deviation > 0 ? deviation : 20));

   if(!sent)
      return StringFormat("{\"ok\":false,\"error\":\"close failed, retcode=%d: %s\"}",
                           trade.ResultRetcode(), trade.ResultRetcodeDescription());

   return StringFormat(
      "{\"ok\":true,\"retcode\":%d,\"ticket\":%I64u,\"price\":%.5f,\"volume\":%.2f,\"comment\":\"%s\"}",
      trade.ResultRetcode(), ticket, trade.ResultPrice(), trade.ResultVolume(), trade.ResultRetcodeDescription());
  }

//+------------------------------------------------------------------+
//| Minimal hand-rolled JSON field extraction. MQL5 has no built-in   |
//| JSON library; this deliberately only supports the FLAT request    |
//| schema above (no nested objects/arrays ever appear in a request), |
//| which keeps this tractable without pulling in a full parser.      |
//+------------------------------------------------------------------+
string ExtractString(string json, string key)
  {
   string pattern = "\"" + key + "\":\"";
   int start = StringFind(json, pattern);
   if(start < 0) return "";
   start += StringLen(pattern);
   int end = StringFind(json, "\"", start);
   if(end < 0) return "";
   return StringSubstr(json, start, end - start);
  }

double ExtractDouble(string json, string key)
  {
   string pattern = "\"" + key + "\":";
   int start = StringFind(json, pattern);
   if(start < 0) return 0.0;
   start += StringLen(pattern);
   int len = StringLen(json);
   int end = start;
   while(end < len)
     {
      ushort ch = StringGetCharacter(json, end);
      if(ch != '-' && ch != '.' && (ch < '0' || ch > '9'))
         break;
      end++;
     }
   return StringToDouble(StringSubstr(json, start, end - start));
  }

long ExtractLong(string json, string key)
  {
   return (long)ExtractDouble(json, key);
  }
//+------------------------------------------------------------------+
