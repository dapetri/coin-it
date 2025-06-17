import http.client
import json

conn = http.client.HTTPSConnection("api-public.sandbox.exchange.coinbase.com")
payload = ""
headers = {"Content-Type": "application/json"}
conn.request("GET", "/api/v3/brokerage/accounts", payload, headers)
res = conn.getresponse()
data = res.read()
print(data.decode("utf-8"))
