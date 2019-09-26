import html
import unicodedata
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

nltk.data.path.append("./nltk_data/")

def extract_id(s):
    regex = re.compile(r'\d*$')
    return regex.findall(s)[0]

def strip_element(s):
    regex = re.compile(r'<[^>]+>')
    s = html.unescape(s)
    s = regex.sub('', s)
    s = s.replace('\\r\\n', '')
    s = s.replace('\t', '')
    s = html.unescape(s)
    s = unicodedata.normalize("NFKD", s)
    return s

def strip_response(r):
    s = ''
    for d in r:
        for k, v in d.items():
            s += strip_element(v)
    return s

# Prepare text: make lowercase, remove punctuation, remove stopwords
stop_words = set(stopwords.words('english'))

def prepare_text(s):
    words = word_tokenize(s)
    words = [word.lower() for word in words if word.isalpha()]
    filtered_sentence = [] 
    for w in words: 
        if w not in stop_words: 
            filtered_sentence.append(w)
    rejoined_sentence = ' '
    rejoined_sentence = rejoined_sentence.join(filtered_sentence)
    return rejoined_sentence

def generate_request_preview(request, num_words):
  request = strip_element(request)
  l = request.split(' ', num_words)
  l = l[0:num_words]
  preview = ' '.join(l)
  preview = preview + '...'
  return preview

# Start with simple unweighted average. Could pre normalise here so
# compare_vectors is quicker when handling user input
def sent2vec(sentence, model):
    words = word_tokenize(sentence)
    # If sentence is empty, need to return a zeroed numpy array of the
    # correct shape
    dimensions = model.wv.vector_size
    if words == []:
        return np.zeros((1, dimensions))
    vocab = list(model.wv.vocab)
    # Need to remove any words that aren't in the vocab
    safe_words = [word for word in words if word in vocab]
    word_vec_list = []
    if safe_words == []:
      return np.zeros((1, dimensions))
    else:
        for word in safe_words:
            word_vec_list.append(model[word])
        return np.mean(word_vec_list, axis=0)

def search_log(query, model, df_lookup):
    words = word_tokenize(query)
    words = [word.lower() for word in words if word.isalpha()]
    rejoined = ' '.join(words)
    query_vec = sent2vec(rejoined, model)
    df_results = df_lookup[['subject', 'request_preview', 'url', 'id']]
    df_results['cosine_similarity'] = df_lookup.apply(lambda x: cosine_similarity(query_vec.reshape(1, -1), x['sentence_embedding'].reshape(1, -1)), axis=1)
    df_results = df_results.sort_values(by=['cosine_similarity'], ascending=False)
    # cast cosine_similarity to string for display
    df_results['cosine_similarity'] = df_results.apply(lambda x: str(x['cosine_similarity']), axis=1)
    return df_results