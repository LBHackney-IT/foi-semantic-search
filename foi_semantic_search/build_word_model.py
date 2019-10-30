import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import gensim
import files_config
import pickle


def main():
    nltk.data.path.append(files_config.nltk_data_path_local)
    nltk.data.path.append(files_config.nltk_data_path_container)

    df = pd.read_pickle(files_config.preprocessed_filepath)

    # Prepare subject field. We'll treat these as single sentences (most are).
    subject_sentences = df['subject_prepared'].tolist()
    tokenized_subjects = []
    for s in subject_sentences:
        tokenized_subjects.append(word_tokenize(s))

    # Prepare request body
    requestbody_sentences = []
    for i in df.index:
        l = df.iloc[i]['requestbody_prepared']
        requestbody_sentences = requestbody_sentences + l
        tokenized_requestbodies = []
    for s in requestbody_sentences:
        tokenized_requestbodies.append(word_tokenize(s))

    # Put tokenized subjects and request bodies together
    all_tokenized = tokenized_subjects + tokenized_requestbodies

    # Save for use elsewhere
    with open(files_config.tokenized_filepath, 'wb') as f:
        pickle.dump(all_tokenized, f)

    # Build the model
    model_word = gensim.models.Word2Vec(all_tokenized, min_count=1, iter=100)
    model_word.save(files_config.word_model_filepath)


if __name__ == "__main__":
    main()
