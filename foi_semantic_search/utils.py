import html
import unicodedata
import re
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import files_config

nltk.data.path.append(files_config.nltk_data_path_local)
nltk.data.path.append(files_config.nltk_data_path_container)


def extract_id(s):
    regex = re.compile(r'\d*$')
    _id = regex.findall(s)[0]
    _id = int(_id)
    return _id


def strip_element(text):
    regex = re.compile(r'<[^>]+>')
    text = html.unescape(text)
    text = regex.sub('', text)
    text = text.replace('\\r\\n', ' ')
    text = text.replace('\t', ' ')
    text = html.unescape(text)
    text = unicodedata.normalize("NFKD", text)
    return text


def strip_response(response):
    text = ''
    for dictionary in response:
        for k, v in sorted(dictionary.items()):
            text += strip_element(v) + ' '
    return text


# Prepare text: make lowercase, remove punctuation, remove stopwords
stop_words = set(stopwords.words('english'))


def prepare_text(text):
    words = word_tokenize(text)
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
    word_list = request.split(' ', num_words)
    word_list = word_list[0:num_words]
    preview = ' '.join(word_list)
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
