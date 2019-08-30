import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import gensim
import functions
import config

nltk.download('stopwords')
nltk.download('punkt')

filepath = config.data_path + config.preprocessed_filename
df = pd.read_pickle(filepath)

# Build word2vec model from FOI subjects
sentences = df['subject_prepared'].tolist()
tokenized_sentences = [[]]
for s in sentences:
    tokenized_sentences.append(word_tokenize(s))

model_word = gensim.models.Word2Vec(tokenized_sentences, min_count=1, iter=100)
model_word.save(config.word_model_filepath)
