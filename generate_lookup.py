import pandas as pd
import functions
import config
import gensim

# Load models
model = gensim.models.Word2Vec.load(config.word_model_filepath)
tfidf = gensim.models.TfidfModel.load(config.tfidf_filepath)

dictionary = gensim.corpora.Dictionary([list(model.wv.vocab.keys())])

df = pd.read_pickle(config.preprocessed_filepath)

df['request_preview'] = df.apply(lambda x: functions.generate_request_preview(x['requestbody'], 25), axis=1)

# Keep only what we need from the dataframe
df = df[['subject','url','subject_prepared', 'requestbody_prepared', 'id', 'request_preview']]

# From requestbody_prepared (a list of sentences) get a single "sentence"
df['requestbody_concatenated'] = df.apply(lambda x: ' '.join(x['requestbody_prepared']), axis=1)

# Put subject and requestbody together as a single "sentence"
df['subject_requestbody'] = df.apply(lambda x: x['subject_prepared']  + ' ' +  x['requestbody_concatenated'], axis=1)

# Generate sentence embeddings
df['sentence_embedding'] = df.apply(lambda x: functions.sent2vec(sentence=x['subject_requestbody'],model=model, dictionary=dictionary, tfidf=tfidf), axis=1)

# Store the dataframe
df.reset_index(drop=True).to_pickle(config.search_lookup_filepath)
