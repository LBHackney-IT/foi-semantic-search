import files
import gensim
import pickle


def main():
    model = gensim.models.Word2Vec.load(files.word_model_filepath)
    dictionary = gensim.corpora.Dictionary([list(model.wv.vocab.keys())])
    with open(files.tokenized_filepath, 'rb') as f:
        tokenized = pickle.load(f)
    corpus = [dictionary.doc2bow(text) for text in tokenized]
    tfidf = gensim.models.TfidfModel(corpus, id2word=dictionary)
    tfidf.save(files.tfidf_filepath)


if __name__ == "__main__":
    main()
