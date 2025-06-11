# Python Example for subscribing to a channel
import json
import os
from websocket import WebSocketApp
from dotenv import load_dotenv

from archive.utlis import sign_with_jwt

# https://docs.cdp.coinbase.com/coinbase-app/docs/trade/ws-overview
# Channel names: https://docs.cdp.coinbase.com/coinbase-app/docs/trade/ws-channels

# Derived from your Coinbase CDP API Key
# SIGNING_KEY: the signing key provided as a part of your API key. Also called the "SECRET KEY"
# API_KEY: the api key provided as a part of your API key. also called the "API KEY NAME"
load_dotenv()
API_KEY = os.getenv("API_KEY")
SIGNING_KEY = os.getenv("SIGNING_KEY").strip() + "\n"

WS_API_URL = "wss://advanced-trade-ws.coinbase.com"


def on_message(ws, message):
    data = json.loads(message)
    ticker = data["events"][0]["tickers"][0]
    print(
        f'{data["timestamp"]} - {ticker["price"]} {ticker["best_bid"]} {ticker["best_ask"]} - {data["sequence_num"]}'
    )


def on_open(ws: WebSocketApp):
    message = {"type": "subscribe", "channel": "ticker", "product_ids": ["BTC-USD"]}
    signed_message = sign_with_jwt(
        message=message, api_key=API_KEY, signing_key=SIGNING_KEY
    )
    ws.send(json.dumps(signed_message))


def main():
    ws = WebSocketApp(WS_API_URL, on_open=on_open, on_message=on_message)
    ws.run_forever()


if __name__ == "__main__":
    main()
