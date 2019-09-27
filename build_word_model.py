import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import gensim
import functions
import config

nltk.download('stopwords')
nltk.download('punkt')

filepath = config.data_path + config.preprocessed_filename
df = pd.read_pickle(filepath)

# Prepare subject field. We'll treat these as single sentences (most are).
subject_sentences = df['subject_prepared'].tolist()
tokenized_subjects = [[]]
for s in subject_sentences:
    tokenized_subjects.append(word_tokenize(s))

# Prepare request body
requestbody_sentences = []
for i in df.index:
  l = df.iloc[i]['requestbody_prepared']
  requestbody_sentences = requestbody_sentences + l
  tokenized_requestbodies = [[]]
for s in requestbody_sentences:
  tokenized_requestbodies.append(word_tokenize(s))

# Put tokenized subjects and request bodies together
all_tokenized = tokenized_subjects + tokenized_requestbodies

# Build the model
model_word = gensim.models.Word2Vec(all_tokenized, min_count=1, iter=100)
model_word.save(config.word_model_filepath)
