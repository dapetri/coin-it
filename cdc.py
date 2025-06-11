import asyncio
from datetime import datetime
import websockets
import json


# https://exchange-docs.crypto.com/exchange/v1/rest-ws/index.html?javascript#ticker-instrument_name
async def subscribe():
    url = "wss://stream.crypto.com/v2/market"
    payload = {
        "id": 1,
        "method": "subscribe",
        "params": {"channels": ["ticker.BTC_USD"]},
    }
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps(payload))

        while True:
            response = await ws.recv()
            data = json.loads(response)
            if data["method"] == "subscribe":
                p = data["result"]["data"][0]["a"]
                t = data["result"]["data"][0]["t"]
                print(f"{datetime.fromtimestamp(t / 1000)} - {p}")
            elif data["method"] == "public/heartbeat":
                payload_heartbeat = {
                    "id": data["id"],
                    "method": "public/respond-heartbeat",
                }
                await ws.send(json.dumps(payload_heartbeat))
            else:
                print(f"Received unhandled response: {data}")
                exit(1)


asyncio.run(subscribe())
