import html
import unicodedata
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import config

nltk.data.path.append(config.nltk_data_path_local)
nltk.data.path.append(config.nltk_data_path_container)


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


# TF-IDF weighted average of the vectors of the input words
def sent2vec(sentence, model, dictionary, tfidf):
    words = word_tokenize(sentence)
    dimensions = model.wv.vector_size
    if words == []:
        return np.zeros((1, dimensions))
    vocab = list(model.wv.vocab)
    # Need to remove any words that aren't in the vocab
    safe_words = [word for word in words if word in vocab]
    # Deduplicate
    safe_words = list(set(safe_words))
    word_vec_list = []
    if safe_words == []:
        return np.zeros((1, dimensions))
    for word in safe_words:
        word_vec_list.append(model[word])
    weighting = tfidf[dictionary.doc2bow(safe_words)]
    # Need to loop through because these lists are ordered differently
    weighted_vec_list = []
    for tup in weighting:
        word = dictionary[tup[0]]
        weight = tup[1]
        weighted_vec = model[word] * weight
        weighted_vec_list.append(weighted_vec)
    return np.mean(weighted_vec_list, axis=0)


def search_log(query, num_results, model, df_lookup, dictionary, tfidf):
    words = word_tokenize(query)
    words = [word.lower() for word in words if word.isalpha()]
    rejoined = ' '.join(words)
    query_vec = sent2vec(rejoined, model, dictionary, tfidf)
    df_results = df_lookup[['subject', 'request_preview', 'url', 'id']]
    df_results['cosine_similarity'] = df_lookup.apply(
        lambda x: cosine_similarity(
            query_vec.reshape(1, -1), x['sentence_embedding'].reshape(1, -1)
        ),
        axis=1,
    )
    df_results = df_results.sort_values(by=['cosine_similarity'], ascending=False)
    # cast cosine_similarity to string for display
    df_results['cosine_similarity'] = df_results.apply(
        lambda x: str(x['cosine_similarity']), axis=1
    )
    df_results = df_results.head(num_results)
    return df_results
