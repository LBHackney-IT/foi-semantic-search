import json
import pandas as pd
from pandas.io.json import json_normalize
from nltk.tokenize import sent_tokenize
import nltk
import utils
import files_config


def prepare_requestbody(s):
    l = sent_tokenize(s)
    l = [utils.prepare_text(e) for e in l]
    l = [e for e in l if e != '']
    return l


def main(input_data="all"):
    # Read json files from infreemation reporting API, new ones added
    # periodically
    # data: "all" or "latest"
    base_path = files_config.raw_data_path
    filename = 'infreemation-dump-'
    df = pd.DataFrame()

    for i in range(1, 6):
        filepath = base_path + filename + str(i) + '.json'
        with open(filepath) as f:
            data = f.read()
        data = json.loads(data)
        dff = json_normalize(data['published']['request'])
        df = df.append(dff)
        df = df.reset_index(drop=True)

    # Need to get the FOI ID from the url field
    for i in df.index:
        df.at[i, 'id'] = utils.extract_id(df.iloc[i]['url'])

    # Prepare subject field, which is plain text
    for i in df.index:
        try:
            df.at[i, 'subject_prepared'] = utils.prepare_text(df.iloc[i]['subject'])
        except:
            print(df.iloc[i]['subject'])

    # Prepare request body
    # Strip HTML
    for i in df.index:
        df.at[i, 'requestbody_stripped'] = utils.strip_element(
            df.iloc[i]['requestbody']
        )
    # Remove stopwords, non alpha, etc.
    df['requestbody_prepared'] = df.apply(
        lambda x: prepare_requestbody(x['requestbody_stripped']), axis=1
    )

    # Store pre-processed data
    filename = files_config.preprocessed_filename
    filepath = base_path + filename
    df.reset_index(drop=True).to_pickle(filepath)


if __name__ == "__main__":
    # handle command line args here
    main()
