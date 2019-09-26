import json
import pandas as pd
from pandas.io.json import json_normalize
from nltk.tokenize import sent_tokenize
import nltk
import functions
import config

def prepare_requestbody(s):
  l = sent_tokenize(s)
  l = [functions.prepare_text(e) for e in l]
  l = [e for e in l if e != '']
  return l

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

# Prepare subject field, which is plain text
for i in df.index:
    try:
        df.at[i, 'subject_prepared'] = functions.prepare_text(df.iloc[i]['subject'])
    except:
        print(df.iloc[i]['subject'])

# Prepare request body
# Strip HTML
for i in df.index:
    df.at[i, 'requestbody_stripped'] = functions.strip_element(s = df.iloc[i]['requestbody'])
# Remove stopwords, non alpha, etc.
df['requestbody_prepared'] = df.apply(lambda x: prepare_requestbody(x['requestbody_stripped']), axis=1)

# Store pre-processed data
filename = config.preprocessed_filename
filepath = base_path + filename
df.reset_index(drop=True).to_pickle(filepath)