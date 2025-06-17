import os
import requests
import time
import hmac
import hashlib
import json
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("CRYPTO_API_KEY")
secret_key = os.getenv("CRYPTO_SECRET_KEY")
url = "https://api.crypto.com/exchange/v1/"
url = "https://uat-api.3ona.co/exchange/v1/"


req = {
    "id": 11,
    "method": "private/user-balance",
    "api_key": api_key,
    "params": {},
    "nonce": int(time.time() * 1000),
}

# First ensure the params are alphabetically sorted by key
param_str = ""

MAX_LEVEL = 3


def params_to_str(obj, level):
    if level >= MAX_LEVEL:
        return str(obj)

    return_str = ""
    for key in sorted(obj):
        return_str += key
        if obj[key] is None:
            return_str += "null"
        elif isinstance(obj[key], list):
            for subObj in obj[key]:
                return_str += params_to_str(subObj, level + 1)
        else:
            return_str += str(obj[key])
    return return_str


if "params" in req:
    param_str = params_to_str(req["params"], 0)


req["sig"] = hmac.new(
    bytes(str(secret_key), "utf-8"),
    msg=bytes(
        req["method"] + str(req["id"]) + req["api_key"] + param_str + str(req["nonce"]),
        "utf-8",
    ),
    digestmod=hashlib.sha256,
).hexdigest()


response = requests.post(
    url + "private/user-balance",
    json=req,
)

print(response.json())
# print(requests.get(url + "/public/get-tickers?instrument_name=BTCUSD-PERP").json())
