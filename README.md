# foi-semantic-search

This is a project undertaken as part of the Government Data Science Accelerator programme. It provides a semantic search capability for the London Borough of Hackney's Freedom of Information disclosure log. The goal is to enable increased accessibility of previously published information. The project is currently based on a `word2vec` model. Sentence/document embeddings are generated as a TF-IDF weighted average. Search functionality is exposed via an API.

## API

The `Dockerfile` builds an image for deployment of the API (using FastAPI). It's configured for Google Cloud Run (HTTP port is set with the `$PORT` environment variable). OpenAPI (Swagger) docs are available at `https://<domain>/docs`

## Dashboard
`dashboard.py` launches a Plotly Dash dashboard with interactive visualizations of the model and a disclosure log search form. This can be deployed to Google App Engine (standard Python 3.7 environment) using the configuration in `app.yaml`.