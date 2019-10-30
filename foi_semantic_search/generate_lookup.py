import pandas as pd
import utils
import files_config
import gensim


def main():
    # Load models
    model = gensim.models.Word2Vec.load(files_config.word_model_filepath)
    tfidf = gensim.models.TfidfModel.load(files_config.tfidf_filepath)

    dictionary = gensim.corpora.Dictionary([list(model.wv.vocab.keys())])

    df = pd.read_pickle(files_config.preprocessed_filepath)

    df['request_preview'] = df.apply(
        lambda x: utils.generate_request_preview(x['requestbody'], 25), axis=1
    )

    # Keep only what we need from the dataframe
    df = df[
        [
            'subject',
            'url',
            'subject_prepared',
            'requestbody_prepared',
            'id',
            'request_preview',
        ]
    ]

    # From requestbody_prepared (a list of sentences) get a single "sentence"
    df['requestbody_concatenated'] = df.apply(
        lambda x: ' '.join(x['requestbody_prepared']), axis=1
    )

    # Put subject and requestbody together as a single "sentence"
    df['subject_requestbody'] = df.apply(
        lambda x: x['subject_prepared'] + ' ' + x['requestbody_concatenated'], axis=1
    )

    # Generate sentence embeddings
    df['sentence_embedding'] = df.apply(
        lambda x: utils.sent2vec(
            sentence=x['subject_requestbody'],
            model=model,
            dictionary=dictionary,
            tfidf=tfidf,
        ),
        axis=1,
    )

    # Store the dataframe
    df.reset_index(drop=True).to_pickle(files_config.search_lookup_filepath)


if __name__ == "__main__":
    main()
