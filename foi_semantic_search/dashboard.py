# -*- coding: utf-8 -*-

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import flask
import pandas as pd
import plotly.graph_objs as go
import utils
import files_config
import plotly.express as px
import gensim
import nltk
from nltk.tokenize import word_tokenize
from dash.exceptions import PreventUpdate
import utils

nltk.data.path.append(files_config.nltk_data_path_local)
nltk.data.path.append(files_config.nltk_data_path_container)

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)

# vocab with dimensionality reduction and most similar already calculated
df = pd.read_pickle(files_config.viz_df_filepath)
# need to index on the vocab but also need to pass plotly a column
# name to label the scatterplot, so will duplicate the vocab field
# until can find a better way
df['index'] = df['vocab']
df = df.set_index('index')

# Word2vec model
model_word = gensim.models.Word2Vec.load(files_config.word_model_filepath)

tfidf = gensim.models.TfidfModel.load(files_config.tfidf_filepath)
dictionary = gensim.corpora.Dictionary([list(model_word.wv.vocab.keys())])

df_subjects = pd.read_pickle(files_config.search_lookup_filepath)

layout_children = [
    html.H2('FOI search/similarity'),
    html.Div(
        "Based on a word2vec model, corpus is request bodies and subject lines from Hackney's disclosure log"
    ),
    dcc.Graph(id="graph", style={"width": "75%", "display": "inline-block"}),
    html.H5('Most similar to:'),
    dcc.Dropdown(
        id='word-dropdown',
        placeholder='Search words in model vocabulary...',
        options=[{'value': i, 'label': i} for i in df.index],
        multi=False,
    ),
    html.Div(id='most-similar'),
    html.Br(),
    html.Hr(),
    html.H5('What do you want to know?'),
    html.Div(
        'Returns suggestions from the disclosure log. Based on cosine similarity of vectors of submitted text vs requests. These sentence/document vectors are a TF-IDF weighted average of the vectors of the constituent words.'
    ),
    html.Br(),
    dcc.Textarea(
        id='search-textarea',
        placeholder='Your request...',
        rows=50,
        style={'width': '50%'},
    ),
    html.Br(),
    html.Button('Submit', id='search_log_button'),
    html.Br(),
    html.Div(id='results-list'),
    html.Br(),
    html.Br(),
    html.Hr(),
]

app.layout = html.Div(children=layout_children)


@app.callback(
    Output('results-list', 'children'),
    [Input('search-textarea', 'value'), Input('search_log_button', 'n_clicks')],
)
def update_search_results(query, n_clicks):
    if not n_clicks:
        return html.Div()
    ctx = dash.callback_context
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if not (ctx.triggered and input_id == 'search_log_button'):
        raise PreventUpdate
    else:
        df_results = utils.search_log(
            query, 5, model_word, df_subjects, dictionary, tfidf
        )
        rows = []
        for i in range(len(df_results)):
            row = []
            for col in df_results.columns:
                value = df_results.iloc[i][col]
                if col == 'url':
                    cell = html.Td(html.A(href=value, target="_blank", children=value))
                else:
                    cell = html.Td(children=value)
                row.append(cell)
            rows.append(html.Tr(row))
        return html.Table(
            # Header
            [html.Tr([html.Th(col) for col in df_results.columns])]
            + rows
        )


@app.callback(Output('most-similar', 'children'), [Input('word-dropdown', 'value')])
def update_most_similar(chosen_word):
    if chosen_word:
        similar = df.loc[chosen_word]['most_similar']
        return html.Table([html.Tr(html.Td(' '.join(map(str, i)))) for i in similar])


@app.callback(Output("graph", "figure"), [Input('word-dropdown', 'value')])
def make_figure(chosen_word):
    fig = px.scatter(data_frame=df, x='x', y='y', text='vocab')
    return fig


if __name__ == '__main__':
    # app.run_server(debug=True)
    app.run_server(debug=False, port=8080)
