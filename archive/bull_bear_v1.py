import asyncio
from collections import deque
from datetime import datetime
import websockets
import json


class CryptoComTrader:
    def __init__(self):
        self._buy = False
        self._ema = None

    # https://exchange-docs.crypto.com/exchange/v1/rest-ws/index.html?javascript#ticker-instrument_name
    async def calc_candles(self):
        url = "wss://stream.crypto.com/v2/market"
        payload = {
            "id": 1,
            "method": "subscribe",
            "params": {
                "channels": [
                    # "ticker.BTC_USD",
                    "candlestick.1m.BTC_USD",
                ]
            },
        }

        def eval(h_old: float, l_old: float, h_new: float, l_new: float):
            # Bull: 1, bear: -1, neutral: 0
            if h_old < h_new and l_old < l_new:
                eval = 1
            elif h_old > h_new and l_old > l_new:
                eval = -1
            else:
                eval = 0
            return eval

        async with websockets.connect(url) as ws:
            await ws.send(json.dumps(payload))

            first_time = True
            q = deque(maxlen=10)
            s = 0

            while True:
                response = await ws.recv()
                data = json.loads(response)
                if data["method"] == "subscribe":
                    data = data["result"]["data"]
                    current_candle = data[-1]
                    if first_time:
                        first_time = False
                        q = deque(
                            [
                                eval(
                                    h_old=float(data[i - 1]["h"]),
                                    l_old=float(data[i - 1]["l"]),
                                    h_new=float(data[i]["h"]),
                                    l_new=float(data[i]["l"]),
                                )
                                for i in range(-11, -1)
                            ],
                            maxlen=10,
                        )
                        last_candle_in_q = data[-2]
                        prev_candle = data[-1]
                        s = sum(q)
                    elif current_candle["t"] != prev_candle["t"]:
                        # prev_candle now contains the richest information about the last interval
                        b = eval(
                            h_old=float(last_candle_in_q["h"]),
                            l_old=float(last_candle_in_q["l"]),
                            h_new=float(prev_candle["h"]),
                            l_new=float(prev_candle["l"]),
                        )
                        s -= q.popleft() - b
                        q.append(b)
                        if s >= 3:
                            self._buy = True
                        else:
                            self._buy = False
                        if s <= -3:
                            print("BEAR", q)
                        elif s >= 3:
                            print("BULL", q)
                        else:
                            print("NEUTRAL", q)
                        # print(last_candle_in_q)
                        # print(prev_candle)
                        last_candle_in_q = prev_candle
                    prev_candle = current_candle

                elif data["method"] == "public/heartbeat":
                    payload_heartbeat = {
                        "id": data["id"],
                        "method": "public/respond-heartbeat",
                    }
                    await ws.send(json.dumps(payload_heartbeat))
                else:
                    print(f"Received unhandled response: {data}")
                    exit(1)

    async def trade(self):
        url = "wss://stream.crypto.com/v2/market"
        payload = {
            "id": 1,
            "method": "subscribe",
            "params": {"channels": ["ticker.BTC_USD"]},
        }
        async with websockets.connect(url) as ws:
            await ws.send(json.dumps(payload))
            cash = 10
            pf = 0
            purchase_price = 0

            while True:
                response = await ws.recv()
                data = json.loads(response)
                if data["method"] == "subscribe":
                    data = data["result"]["data"][0]
                    p = float(data["a"])
                    best_ask = float(data["k"])
                    best_bid = float(data["b"])
                    self._ema = (
                        0.1 * p + 0.9 * self._ema if self._ema is not None else p
                    )
                    if self._buy and cash and self._ema > best_ask:
                        pf = cash / best_ask
                        purchase_price = best_ask
                        cash = 0
                    if (
                        not self._buy
                        and pf
                        or self._ema < best_bid
                        and purchase_price < best_bid
                        and pf
                    ):
                        cash = pf * best_bid
                        pf = 0

                    print(f"{self._ema - p:.2f}", self._buy, cash, pf)
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
    cc = CryptoComTrader()
    await asyncio.gather(cc.trade(), cc.calc_candles())


if __name__ == "__main__":
    asyncio.run(main())
