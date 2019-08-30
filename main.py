# -*- coding: utf-8 -*-

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import flask
import pandas as pd
import plotly.graph_objs as go
import gensim
import functions
import config
from sklearn.manifold import TSNE
import plotly.express as px
from textwrap import dedent

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)

# Word2vec model
model_word = gensim.models.Word2Vec.load(config.word_model_filepath)
vocab = list(model_word.wv.vocab)

# Reduce dimensions for plotting
X = model_word[vocab]
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
df_vz = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])

df_vz = df_vz.reset_index()

layout_children = [
    html.H2('FOI search/similarity'),
    html.Div("Word2vec model, corpus is request subject lines from Hackney's disclosure log"),
    dcc.Graph(id="graph", style={"width": "75%", "display": "inline-block"}),
    html.Br(),
    html.H5('Most similar to:'),
    dcc.Dropdown(
        id='word-dropdown',
        placeholder='Search words in model vocabulary...',
        options=[{'value': i, 'label': i} for i in vocab],
        multi=False
        ),
    html.Div(id='most-similar')
]

app.layout = html.Div(children=layout_children)

@app.callback(
    Output('most-similar', 'children'),
    [Input('word-dropdown', 'value')]
)
def update_most_similar(chosen_word):
    if chosen_word:
        similar = model_word.wv.most_similar(chosen_word)
        return html.Table([html.Tr(html.Td(' '.join(map(str, i)))) for i in similar])

@app.callback(
    Output("graph", "figure"),
    [Input('word-dropdown', 'value')]
)
def make_figure(chosen_word):
    inds = []
    inds.append(chosen_word)
    fig = px.scatter(data_frame=df_vz, x='x', y='y', text='index')
    #fig.data[0].update(selectedpoints=inds,selected=dict(marker=dict(color='red')),unselected=dict(marker=dict(color='blue')))
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run_server(debug=False, port=8080)
