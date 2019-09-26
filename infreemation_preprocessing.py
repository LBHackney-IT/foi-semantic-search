import json
import pandas as pd
from pandas.io.json import json_normalize
import nltk
import functions
import config


# Read json files from infreemation reporting API, new ones added
# periodically
base_path = config.data_path
filename = 'infreemation-dump-'
df = pd.DataFrame()

for i in range(1,6):
    filepath = base_path + filename + str(i) + '.json'
    with open(filepath) as f:
        data = f.read()
    data = json.loads(data)
    dff = json_normalize(data['published']['request'])
    df = df.append(dff)
    df = df.reset_index(drop=True)

# Need to get the FOI ID from the url field
for i in df.index:
  df.at[i, 'id'] = functions.extract_id(df.iloc[i]['url'])

# Strip HTML from requestbody
for i in df.index:
    df.at[i, 'requestbody_stripped'] = functions.strip_element(s = df.iloc[i]['requestbody'])

# Prepare subject field, which is plain text
for i in df.index:
    try:
        df.at[i, 'subject_prepared'] = prepare_text(df.iloc[i]['subject'])
    except:
        print(df.iloc[i]['subject'])

# Store pre-processed data
filename = config.preprocessed_filename
filepath = base_path + filename
df.reset_index(drop=True).to_pickle(filepath)