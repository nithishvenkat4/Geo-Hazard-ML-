import requests
import pandas as pd
from io import StringIO

print("Downloading INFORM Risk Index...")

url = "https://drmkc.jrc.ec.europa.eu/inform-index/Portals/0/InfoRM/2024/INFORM2024_TREND_2013_2024.csv"

try:

    response = requests.get(url, timeout=60)
    if response.status_code == 200:
        df = pd.read_csv(StringIO(response.text), encoding='latin1')
        df.to_csv("inform_risk.csv", index=False)
        print(f"Done! Shape: {df.shape}")
        print(df.head())
    else:
        print(f"Status code: {response.status_code}")
except Exception as e:
    print(f"Error: {e}")