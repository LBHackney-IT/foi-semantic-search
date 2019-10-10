import requests
import json
import config
from dotenv import load_dotenv
import os

# Download new entries in the FOI disclosure log using the Infreemation
# reporting API. This needs to be run via a whitelisted IP address.
# The API seems to want the params passed in both the URL query string
# and the posted payload.
# Expected date format is YYYY-MM-DD

load_dotenv()
INFREEMATION_USERNAME = os.getenv('INFREEMATION_USERNAME')
INFREEMATION_KEY = os.getenv('INFREEMATION_KEY')

