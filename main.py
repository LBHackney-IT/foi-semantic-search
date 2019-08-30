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

server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server)

df = pd.read_pickle(config.viz_df_filename)
# need to index on the vocab but also need to pass plotly a column
# name to label the scatterplot, so will duplicate the vocab field
# until can find a better way
df['index'] = df['vocab']
df = df.set_index('index')

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
    html.Div(id='most-similar')
]

app.layout = html.Div(children=layout_children)

@app.callback(
    Output('most-similar', 'children'),
    [Input('word-dropdown', 'value')]
)
def update_most_similar(chosen_word):
    if chosen_word:
        #similar = model_word.wv.most_similar(chosen_word)
        similar = df.loc[chosen_word]['most_similar']
        return html.Table([html.Tr(html.Td(' '.join(map(str, i)))) for i in similar])

@app.callback(
    Output("graph", "figure"),
    [Input('word-dropdown', 'value')]
)
def make_figure(chosen_word):
    inds = []
    inds.append(chosen_word)
    fig = px.scatter(data_frame=df, x='x', y='y', text='vocab')
    #fig.data[0].update(selectedpoints=inds,selected=dict(marker=dict(color='red')),unselected=dict(marker=dict(color='blue')))
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run_server(debug=False, port=8080)
