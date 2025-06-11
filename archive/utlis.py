from hashlib import sha256
from os import urandom
from time import time
from jwt import encode


def sign_with_jwt(message, api_key, signing_key):
    payload = {
        "iss": "coinbase-cloud",
        "nbf": int(time()),
        "exp": int(time()) + 120,
        "sub": api_key,
    }
    headers = {"kid": api_key, "nonce": sha256(urandom(16)).hexdigest()}
    token = encode(payload, signing_key, algorithm="ES256", headers=headers)
    message["jwt"] = token
    return message
