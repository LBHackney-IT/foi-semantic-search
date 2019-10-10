from fastapi import FastAPI, Query, Body
import functions
import config
import gensim
import pandas as pd
from pydantic import BaseModel, Schema

app = FastAPI(
    title="Hackney FOI semantic search API",
    description="Search the London Borough of Hackney's public Freedom of Information disclosure log.",
    version="1",
)

# Need this to accept request bodies
class FoiQuery(BaseModel):
    query: str
    results: int = Schema(default=5, description='Number of results to return', max=50)

model = gensim.models.Word2Vec.load(config.word_model_filepath)
tfidf = gensim.models.TfidfModel.load(config.tfidf_filepath)
dictionary = gensim.corpora.Dictionary([list(model.wv.vocab.keys())])
df_lookup = pd.read_pickle(config.search_lookup_filepath)

@app.get('/fois/search/')
def search_fois(
        q: str = Query(
            default=None,
            title='FOI search query',
            description='Can be a simple keyword-style query, a sentence, or a complete FOI request consiting of multiple sentences. Of course this is limited by max URL length. Post your query in a request body instead to avoid this limitation.',
        ),
        results: int = Query(
            default=5,
            description='Number of results to return.',
            max=50,
        )
    ):
    df_results = functions.search_log(q, results, model, df_lookup, dictionary, tfidf)
    results = df_results.to_dict(orient='records')
    return results

@app.post('/fois/search/')
def search_fois_post(foi_query: FoiQuery = Body(..., example = {'query': 'What is the cost of biscuits at all councilor meetings in 2017?', 'results': 5,})):
    df_results = functions.search_log(foi_query.query, foi_query.results, model, df_lookup, dictionary, tfidf)
    results = df_results.to_dict(orient='records')
    return results