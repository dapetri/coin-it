from datetime import datetime
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
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps(payload))

        while True:
            response = await ws.recv()
            data = json.loads(response)
            if data["channel"] == "ticker":
                global p_coinbase, ts_coinbase
                ts_coinbase = data["events"][0]["tickers"][0]["price"]
                p_coinbase = datetime.fromtimestamp(data["timestamp"])
            else:
                print(f"Received unhandled response: {data}")
                exit(1)


# https://exchange-docs.crypto.com/exchange/v1/rest-ws/index.html?javascript#ticker-instrument_name
async def sub_crypto():
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
                global p_crypto, ts_crypto
                p_crypto = data["result"]["data"][0]["a"]
                ts_crypto = data["result"]["data"][0]["t"]
            elif data["method"] == "public/heartbeat":
                payload_heartbeat = {
                    "id": data["id"],
                    "method": "public/respond-heartbeat",
                }
                await ws.send(json.dumps(payload_heartbeat))
            else:
                print(f"Received unhandled response: {data}")
                exit(1)


async def main():
    asyncio.create_task(sub_crypto())
    asyncio.create_task(sub_coinbase())
    time.sleep(3)
    while True:
        print(f"{ts_crypto - ts_coinbase} - {p_crypto} - {p_coinbase}")
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    asyncio.run(main())
