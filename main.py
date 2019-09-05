# -*- coding: utf-8 -*-

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import flask
import pandas as pd
import plotly.graph_objs as go
import functions
import config
import plotly.express as px
import gensim
import nltk
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from dash.exceptions import PreventUpdate


server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)

df = pd.read_pickle(config.viz_df_filename)
# need to index on the vocab but also need to pass plotly a column
# name to label the scatterplot, so will duplicate the vocab field
# until can find a better way
df['index'] = df['vocab']
df = df.set_index('index')

# Word2vec model
model_word = gensim.models.Word2Vec.load(config.word_model_filepath)

df_subjects = pd.read_pickle(config.subjects_lookup_df_filename)

layout_children = [
    html.H2('FOI search/similarity'),
    html.Div("Word2vec model, corpus is request subject lines from Hackney's disclosure log"),
    dcc.Graph(id="graph", style={"width": "75%", "display": "inline-block"}),
    html.Br(),
    html.H5('Most similar to:'),
    dcc.Dropdown(
        id='word-dropdown',
        placeholder='Search words in model vocabulary...',
        options=[{'value': i, 'label': i} for i in df.index],
        multi=False
        ),
    html.Div(id='most-similar'),
    html.Br(),
    html.H5('What do you want to know?'),
    dcc.Textarea(
        id='search-textarea',
        placeholder='Your request...'    
    ),
    html.Br(),
    html.Button('Search disclosure log', id='search_log_button'),
    html.Div(id='results-list'),
]

app.layout = html.Div(children=layout_children)

@app.callback(
    Output('results-list', 'children'),
    [Input('search-textarea', 'value'),
     Input('search_log_button', 'n_clicks')]
)
def update_search_results(query, n_clicks):
    if not n_clicks:
        return html.Div()
    ctx = dash.callback_context
    input_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if not (ctx.triggered and input_id == 'search_log_button'):
        raise PreventUpdate
    else:
        words = word_tokenize(query)
        words = [word.lower() for word in words if word.isalpha()]
        rejoined = ' '.join(words)
        query_vec = functions.sent2vec(rejoined, model_word)
        df_results = df_subjects[['subject', 'request_preview', 'url']]
        df_results['cosine_similarity'] = df_subjects.apply(lambda x: cosine_similarity(query_vec.reshape(1, -1), x['subject_embedding'].reshape(1, -1)), axis=1)
        df_results = df_results.sort_values(by=['cosine_similarity'])
        df_results = df_results[['subject', 'request_preview', 'url']].tail()
        rows = []
        for i in range(len(df_results)):
            row = []
            for col in df_results.columns:
                value = df_results.iloc[i][col]
                if col == 'url':
                    cell = html.Td(html.A(href=value, children=value))
                else:
                    cell = html.Td(children=value)
                row.append(cell)
            rows.append(html.Tr(row))
        return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in df_results.columns])] +
        rows
        )

@app.callback(
    Output('most-similar', 'children'),
    [Input('word-dropdown', 'value')]
)
def update_most_similar(chosen_word):
    if chosen_word:
        similar = df.loc[chosen_word]['most_similar']
        return html.Table([html.Tr(html.Td(' '.join(map(str, i)))) for i in similar])

@app.callback(
    Output("graph", "figure"),
    [Input('word-dropdown', 'value')]
)
def make_figure(chosen_word):
    fig = px.scatter(data_frame=df, x='x', y='y', text='vocab')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run_server(debug=False, port=8080)
