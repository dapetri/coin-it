import json
from time import sleep
from os import getenv
from coinbase.websocket import WSClient
from dotenv import load_dotenv

load_dotenv(".env")
API_KEY = getenv("API_KEY")
SIGNING_KEY = getenv("SIGNING_KEY") + "\n"
WS_API_URL = "wss://advanced-trade-ws.coinbase.com"


def on_message(msg):
    data = json.loads(msg)
    if "ticker" == data["channel"]:
        ticker = data["events"][0]["tickers"][0]
        print(
            f'{data["timestamp"]} - {ticker["price"]} {ticker["best_bid"]} {ticker["best_ask"]} - {data["sequence_num"]}'
        )


# https://github.com/coinbase/coinbase-advanced-py/
client = WSClient(
    base_url=WS_API_URL,
    api_key=API_KEY,
    api_secret=SIGNING_KEY,
    on_message=on_message,
)


client.open()
client.ticker(product_ids=["BTC-USD", "ETH-USD"])

# wait 10 seconds
sleep(3)

client.ticker_unsubscribe(product_ids=["BTC-USD", "ETH-USD"])
client.close()
