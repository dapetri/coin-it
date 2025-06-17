from collections import deque
from datetime import datetime, timezone
import time
import json
import asyncio
import websockets
import json

p_crypto = None
ts_crypto = None
p_coinbase = None
ts_coinbase = None


# https://docs.cdp.coinbase.com/coinbase-app/docs/trade/ws-auth
async def sub_coinbase():
    url = "wss://advanced-trade-ws.coinbase.com"
    payload = {
        "type": "subscribe",
        "product_ids": ["BTC-USD"],
        "channel": "ticker",
    }
    n = 40
    q = deque(maxlen=n)
    s = 0.0
    win = 0
    dep = 0
    cash = 10
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps(payload))

        while True:
            response = await ws.recv()
            data = json.loads(response)
            if data["channel"] == "ticker":
                p = float(data["events"][0]["tickers"][0]["price"])
                s += p
                if len(q) == n:
                    s -= q[0]
                q.append(p)
                if len(q) < n:
                    print(len(q))
                    continue
                sma = round(s / n, 2)
                if p < sma and cash:
                    dep += cash / p
                    cash = 0
                if p > sma and dep:
                    cash += p * dep
                    dep = 0
                if cash > 10:
                    win += cash - 10
                    cash = 10
                print(f"Price: {p}, SMA: {sma}, Cash: {cash}, Dep: {dep}, Win: {win}")
            elif data["channel"] == "subscriptions":
                pass
            else:
                print(f"Received unhandled response: {data}")
                exit(1)


if __name__ == "__main__":
    asyncio.run(sub_coinbase())
