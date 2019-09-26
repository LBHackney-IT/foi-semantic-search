import sys
import pytest
from synthetic_queries import queries
import gensim
import pandas as pd
import functions
import config

# put into fixture?
model_word = gensim.models.Word2Vec.load(config.word_model_filepath)
df_subjects = pd.read_pickle(config.subjects_lookup_df_filename)

# Pass synthetic queries to search function; the expected
# result document should appear in the top n results
@pytest.mark.parametrize("input_query,expected_ID", queries)
def test_eval(input_query, expected_ID):
    results = functions.search_log(input_query, model_word, df_subjects)
    candidates = results['id'].head(5).tolist()
    assert expected_ID in candidates