import pandas as pd
import functions
import config
import gensim

# Word2vec model
model_word = gensim.models.Word2Vec.load(config.word_model_filepath)

base_path = ''

filepath = config.data_path + config.preprocessed_filename
df = pd.read_pickle(filepath)

# Keep only what we need from the dataframe
df = df[['subject','url','subject_prepared','id']]

# Generate sentence embeddings
df['subject_embedding'] = df.apply(lambda x: functions.sent2vec(sentence=x['subject_prepared'],model=model_word), axis=1)

# Store the dataframe
filename = config.minimal_df_filename
filepath = base_path + filename
df.reset_index(drop=True).to_pickle(filepath)

