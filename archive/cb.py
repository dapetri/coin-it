import json
import asyncio
import websockets
import json


# https://docs.cdp.coinbase.com/coinbase-app/docs/trade/ws-auth
async def subscribe_cb():
    url = "wss://advanced-trade-ws.coinbase.com"
    payload = {
        "type": "subscribe",
        "product_ids": ["BTC-USD"],
        "channel": "ticker",
    }
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps(payload))

        while True:
            response = await ws.recv()
            data = json.loads(response)
            if data["channel"] == "ticker":
                p = data["events"][0]["tickers"][0]["price"]
                t = data["timestamp"]
                print(f"{t} - {p}")
            else:
                print(f"Received unhandled response: {data}")
                exit(1)


asyncio.run(subscribe_cb())
