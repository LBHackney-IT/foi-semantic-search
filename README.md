# foi-model

This is a project undertaken as part of the Government Data Science Accelerator programme. It provides a semantic search capability for the London Borough of Hackney's Freedom of Information disclosure log. I expect to add document similarity/"more like this" functionality in the near fuure.

## API

The search functionality is exposed via an API. The main `Dockerfile` will build an image intended for deployment of the API. It's configured for Google Cloud Run (allowing the HTTP port to be set with a `$PORT` environment variable).

## Dashboard
`dashboard.py` launches a Plotly Dash dashboard with interactive visualizations of the model and a disclosure log search form. This can be deployed to Google App Engine (standard Python 3.7 environment). App Engine configuration is in `app.yaml`.

## Incorporating new requests

Newly published requests are downloaded from the hosted FOI disclosure log via a reporting API.

Secrets are managed using `python-dotenv`.

## Preprocessing

## Model training

## Generating the search lookup

## Deployment

## Running tests