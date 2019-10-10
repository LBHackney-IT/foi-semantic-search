import config
import gensim
import pickle

model = gensim.models.Word2Vec.load(config.word_model_filepath)
dictionary = gensim.corpora.Dictionary([list(model.wv.vocab.keys())])
with open(config.tokenized_filepath, 'rb') as f:
    tokenized = pickle.load(f)
corpus = [dictionary.doc2bow(text) for text in tokenized]
tfidf = gensim.models.TfidfModel(corpus, id2word=dictionary)
tfidf.save(config.tfidf_filepath)
