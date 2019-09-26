import pandas as pd
import functions
import config
import gensim

# Word2vec model
model_word = gensim.models.Word2Vec.load(config.word_model_filepath)

df = pd.read_pickle(config.preprocessed_filepath)

df['request_preview'] = df.apply(lambda x: functions.generate_request_preview(x['requestbody'], 25), axis=1)

# Keep only what we need from the dataframe
df = df[['subject','url','subject_prepared', 'requestbody_prepared', 'id', 'request_preview']]


# Generate sentence embeddings
df['sentence_embedding'] = df.apply(lambda x: functions.sent2vec(sentence=x['subject_requestbody'],model=model_word), axis=1)

# Store the dataframe
df.reset_index(drop=True).to_pickle(config.search_lookup_filepath)
