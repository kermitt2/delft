import os
import json
from utilities.Embeddings import make_embeddings_simple
from utilities.Utilities import clean_text
import textClassification

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def make_df(train_path, test_path, max_features, maxlen, list_classes):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df.comment_text.fillna('MISSINGVALUE', inplace=True)
    test_df.comment_text.fillna('MISSINGVALUE', inplace=True)
    
    train_df['comment_text'] = train_df['comment_text'].apply(lambda x: clean_text(str(x)))
    test_df['comment_text'] = test_df['comment_text'].apply(lambda x: clean_text(str(x)))

    train_df = train_df.sample(frac=1)
    
    #print('train_df.shape:', train_df.shape)
    #print('train_df:', train_df)
    
    list_sentences_train = train_df["comment_text"].values
    y = train_df[list_classes].values
    list_sentences_test = test_df["comment_text"].values

    list_sentences_all = np.concatenate([list_sentences_train, list_sentences_test])

    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_all))
    print('word_index size:', len(tokenizer.word_index))

    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    
    X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)
    word_index = tokenizer.word_index

    return X_t, X_te, y, word_index


def train(embedding_vector): 
    model = textClassification.Classifier('toxic', "gru", max_epoch=25, embeddings=embedding_vector, word_emb_size=embed_size)

    train_df = pd.read_csv(train_path) 
    train_df.comment_text.fillna('MISSINGVALUE', inplace=True)

    model.train(xtr, y)

    # saving the model
    model.save()


def train_and_test(embedding_vector): 
    model = textClassification.Classifier('toxic', "gru", max_epoch=25, embeddings=embedding_vector, word_emb_size=embed_size)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    train_df.comment_text.fillna('MISSINGVALUE', inplace=True)
    test_df.comment_text.fillna('MISSINGVALUE', inplace=True)


    model.train(x_train, y_train)

    model.eval(x_test, y_test)

    # saving the model
    model.save()

# classify a list of texts
def classify(texts, embedding_vector):
    # load model
    model = textClassification.Classifier('toxic', "gru", embeddings=embedding_vector)
    model.load()
    results = []

    for text in texts:
        cleaned = clean_text(text)
        results.append(model.predict(text))

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Classification of comments/short texts in toxicity types (toxic, severe_toxic, obscene, threat, insult, identity_hate)")

    parser.add_argument("action")
    parser.add_argument("--fold-count", type=int, default=1)

    args = parser.parse_args()

    print('importing embeddings...')
    embedding_vector = make_embeddings_with_oov("/mnt/data/wikipedia/embeddings/crawl-300d-2M.vec", "/mnt/data/mimick/oov.fastText.embeddings.vec",
                                    max_features, embed_size, word_index, True)
    print("embedding size:", embedding_vector.shape)

    file_path = modelName+".model.hdf5"

    print('building dataframe...')
    #xtr, xte, y, word_index = make_df("../data/textClassification/toxic/train.csv",
    #                                "../data/textClassification/toxic/test.csv",
    #                                max_features, maxlen, list_classes) 
    if action == 'train':
        if args.fold_count < 1:
            raise ValueError("fold-count should be equal or more than 1")
        train(embedding_vector)

    if action == 'test':
        y_test = test(embedding_vector)    

        # write test predictions as a submission file 
        sample_submission = pd.read_csv("../data/textClassification/toxic/sample_submission.csv")
        sample_submission[list_classes] = y_test
        sample_submission.to_csv("../../data/textClassification/toxic/"+resultFile, index=False)

    if action == 'classify':
        someTexts = ['This is a gentle test.', 'This is a fucking test!', 'This is a test and I know where you leave.']
        classify(someTexts, embedding_vector)

    # see https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()
