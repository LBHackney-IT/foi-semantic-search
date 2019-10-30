import sys
import pytest
from test_data import queries
import gensim
import pandas as pd
from foi_model import utils
from foi_model import files_config

# put into fixture?
model = gensim.models.Word2Vec.load(files_config.word_model_filepath)
tfidf = gensim.models.TfidfModel.load(files_config.tfidf_filepath)
dictionary = gensim.corpora.Dictionary([list(model.wv.vocab.keys())])
df_lookup = pd.read_pickle(files_config.search_lookup_filepath)

# Pass synthetic queries to search function; the expected
# result document should appear in the top n results
@pytest.mark.parametrize("input_query,expected_ID", queries)
def test_eval(input_query, expected_ID):
    results = utils.search_log(input_query, 5, model, df_lookup, dictionary, tfidf)
    candidates = results['id'].tolist()
    assert expected_ID in candidates
