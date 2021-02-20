from pprint import pprint
import requests

#url = "https://tech-diary.net"

#r = requests.get(url)
# print(r.text)


# API
#url = "https://coincheck.com/api/ticker"
#r = requests.get(url)
# print(r.json())

BASE_URL = "https://coincheck.com"
url = BASE_URL + "/api/trades"

params = {
    "pair": "btc_jpy"
}

r = requests.get(url, params=params)

if "json" in r.headers.get("content-type"):
    r = r.json()
    pprint(r)
else:
    r = r.text
    pprint(r)
