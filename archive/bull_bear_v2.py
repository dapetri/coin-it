import asyncio
from collections import deque
from datetime import datetime
import websockets
import json


class CryptoComTrader:
    def __init__(self):
        self._buy = False

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

        def eval(
            h_old: float,
            l_old: float,
            c_old: float,
            o_old: float,
            h_new: float,
            l_new: float,
            c_new: float,
            o_new: float,
        ):
            if c_new > c_old and o_new > o_old:
                trend = 1
            elif c_new < c_old and o_new < o_old:
                trend = -1
            else:
                trend = 0
            # csv = c_new - o_new / (h_new - l_new)
            return trend

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
                                    c_old=float(data[i - 1]["c"]),
                                    o_old=float(data[i - 1]["o"]),
                                    h_new=float(data[i]["h"]),
                                    l_new=float(data[i]["l"]),
                                    c_new=float(data[i]["c"]),
                                    o_new=float(data[i]["o"]),
                                )
                                for i in range(-6, -1)
                            ],
                        )
                        last_candle_in_q = data[-2]
                        prev_candle = data[-1]
                        s = sum(q)
                    elif current_candle["t"] != prev_candle["t"]:
                        # prev_candle now contains the richest information about the last interval
                        b = eval(
                            h_old=float(last_candle_in_q["h"]),
                            l_old=float(last_candle_in_q["l"]),
                            c_old=float(last_candle_in_q["c"]),
                            o_old=float(last_candle_in_q["o"]),
                            h_new=float(prev_candle["h"]),
                            l_new=float(prev_candle["l"]),
                            c_new=float(prev_candle["c"]),
                            o_new=float(prev_candle["o"]),
                        )
                        s -= q.popleft() - b
                        q.append(b)
                        if s >= 2:
                            self._buy = True
                        else:
                            self._buy = False
                        last_candle_in_q = prev_candle
                        print(
                            q,
                            "\n",
                            end="\r",
                            flush=True,
                        )
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
            cash = 100
            pf = 0
            buy_price = None

            while True:
                response = await ws.recv()
                data = json.loads(response)
                if data["method"] == "subscribe":
                    data = data["result"]["data"][0]
                    p = float(data["a"])
                    best_ask = float(data["k"])
                    best_bid = float(data["b"])
                    if self._buy and cash:
                        pf, cash = cash / best_ask, 0
                        buy_price = best_ask
                    if not self._buy and pf or buy_price and best_bid < buy_price:
                        cash, pf = pf * best_bid, 0
                    print(
                        "Buy:",
                        self._buy,
                        f"- {cash:.2f} USD",
                        f"- {pf} units",
                        end="\r",
                        flush=True,
                    )
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
