from datetime import datetime, timezone
import time
import json
import asyncio
import websockets
import json

crypto_ticker = None
coinbase_ticker = None


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
                global coinbase_ticker
                coinbase_ticker = data["events"][0]["tickers"][0]
            elif data["channel"] == "subscriptions":
                pass
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
                global crypto_ticker
                crypto_ticker = data["result"]["data"][0]
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
    while not crypto_ticker or not coinbase_ticker:
        await asyncio.sleep(1)
    while True:
        spread_crypto = float(crypto_ticker["b"]) - float(coinbase_ticker["best_ask"])
        spread_coinbase = float(coinbase_ticker["best_bid"]) - float(crypto_ticker["k"])
        print(
            f"Spread Crypto: {spread_crypto:.2f} | " f"Spread CB: {spread_coinbase:.2f}"
        )
        await asyncio.sleep(0.3)


if __name__ == "__main__":
    asyncio.run(main())
