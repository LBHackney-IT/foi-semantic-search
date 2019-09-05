import html
import unicodedata
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize

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
