from fastapi import FastAPI
import functions
import config
import gensim
import pandas as pd
from pydantic import BaseModel

app = FastAPI()
app = FastAPI(
    title="Hackney FOI semantic search API",
    description="Search the London Borough of Hackney's public Freedom of Information disclosure log.",
    version="1",
)

class FoiQuery(BaseModel):
    query: str

model = gensim.models.Word2Vec.load(config.word_model_filepath)
tfidf = gensim.models.TfidfModel.load(config.tfidf_filepath)
dictionary = gensim.corpora.Dictionary([list(model.wv.vocab.keys())])

df_lookup = pd.read_pickle(config.search_lookup_filepath)

@app.get('/foi/search/{query}')
def foi_search_get(query: str):
    df_results = functions.search_log(query, model, df_lookup, dictionary, tfidf)
    results = df_results.head().to_dict(orient='records')
    return results

@app.post('/foi/search/')
def foi_search_post(foi_query: FoiQuery):
    df_results = functions.search_log(foi_query.query, model, df_lookup, dictionary, tfidf)
    results = df_results.head().to_dict(orient='records')
    return results