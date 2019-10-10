import gensim
import pandas as pd
from sklearn.manifold import TSNE
import config

# Preprocessing for the dashboard so it starts up more quickly and
# consumes less memory -- these factors are important for deployment on
# App Engine

# Word2vec model
model_word = gensim.models.Word2Vec.load(config.word_model_filepath)
vocab = list(model_word.wv.vocab)

# Reduce vector dimensions for plotting
X = model_word[vocab]
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
df = df.reset_index()
df.columns = ['vocab', 'x', 'y']
df = df.sort_values(by=['vocab'])

# Cache most similar words for all words in vocab
df['most_similar'] = df.apply(lambda x: model_word.wv.most_similar(x['vocab']), axis=1)

df.to_pickle(config.viz_df_filepath)
