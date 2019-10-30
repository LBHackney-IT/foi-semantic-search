import requests
import json
import files_config
from dotenv import load_dotenv
import os
import pandas as pd
from datetime import date, datetime as dt
from shutil import copyfile

# ----------------------------------------------------------------------
# Download new entries in the FOI disclosure log using the Infreemation
# reporting API. This needs to be run via a whitelisted IP address.
# Expected date format is YYYY-MM-DD
# ----------------------------------------------------------------------

# Get our preprocessed requests
df = pd.read_pickle(files_config.preprocessed_filepath)
# Get date of latest request we already have
latest_date = df['datepublished'].sort_values(ascending=False).iloc[0]

today_date = dt.strftime(date.today(), '%Y-%m-%d')

# Get secrets for API
load_dotenv()
INFREEMATION_USERNAME = os.getenv('INFREEMATION_USERNAME')
INFREEMATION_KEY = os.getenv('INFREEMATION_KEY')

# Ask for published requests from that date until today. The API seems
# to want the params passed in both the URL query string and the posted
# payload.
url = f'https://api.infreemation.co.uk/live/foi/?rt=PUBLISHED&key={INFREEMATION_KEY}&username={INFREEMATION_USERNAME}&startdate={latest_date}&enddate={today_date}&status=all'

body = {
    "rt": "PUBLISHED",
    "key": INFREEMATION_KEY,
    "username": INFREEMATION_USERNAME,
    "startdate": latest_date,
    "enddate": today_date,
    "status": "all",
}

response = requests.post(url, data=json.dumps(body))

# Write an archive version of the file to be used when model training
# from scratch
filename = f'infreemation-download-{latest_date}-to-{today_date}-retrieved-on-{today_date}.json'
archive_filepath = files_config.raw_data_path + filename
print('writing ' + archive_filepath)
with open(archive_filepath, 'wb') as f:
    f.write(response.content)

# Make the copy that gets picked up when updating the model and the
# search lookup
latest_filepath = files_config.raw_data_path + 'latest.json'
print('copying to ' + latest_filepath)
copyfile(archive_filepath, latest_filepath)
