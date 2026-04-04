import requests
import pandas as pd
import time

print("Starting earthquake data download in chunks...")

all_data = []

# Download year by year to stay under the 20,000 limit
years = list(range(2000, 2024))

for year in years:
    start = f"{year}-01-01"
    end   = f"{year}-12-31"
    
    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query.csv"
        f"?starttime={start}&endtime={end}"
        "&minmagnitude=4.5"
        "&orderby=time"
        "&limit=20000"
    )
    
    print(f"Downloading {year}...", end=" ")
    
    try:
        response = requests.get(url, timeout=60)
        
        if response.status_code == 200:
            from io import StringIO
            df_year = pd.read_csv(StringIO(response.text))
            all_data.append(df_year)
            print(f"{len(df_year)} events downloaded")
        else:
            print(f"Failed - status {response.status_code}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    # Be polite to the server - wait 1 second between requests
    time.sleep(1)

# Combine all years into one dataframe
print("\nCombining all years...")
earthquakes = pd.concat(all_data, ignore_index=True)

# Save to your project folder
earthquakes.to_csv("earthquakes.csv", index=False)

print(f"\nDone! Total events: {len(earthquakes)}")
print("File saved as earthquakes.csv")
print(earthquakes[['time','latitude','longitude','mag','depth']].head())

#python download_data.py