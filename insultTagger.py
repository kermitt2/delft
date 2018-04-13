import os
import json
from utilities.Embeddings import make_embeddings_simple
import sequenceLabelling
from sequenceLabelling.tokenizer import tokenizeAndFilter
from sequenceLabelling.reader import load_data_and_labels_xml_file, load_data_and_labels_conll


def train(): 
    root = os.path.join(os.path.dirname(__file__), 'data/sequenceLabelling/toxic/')

    train_path = os.path.join(root, 'corrected.xml')
    valid_path = os.path.join(root, 'valid.xml')

    print('Loading data...')
    x_train, y_train = load_data_and_labels_xml_file(train_path)
    x_valid, y_valid = load_data_and_labels_xml_file(valid_path)
    print(len(x_train), 'train sequences')
    print(len(x_valid), 'validation sequences')

    embed_size, embedding_vector = make_embeddings_simple("/mnt/data/wikipedia/embeddings/crawl-300d-2M.vec")

    model = sequenceLabelling.Sequence('insult', max_epoch=50, embeddings=embedding_vector)
    model.train(x_train, y_train, x_valid, y_valid)
    print('training done')

    tokens = tokenizeAndFilter('you\'re a moronic wimp who is too lazy to do research! die in hell !!')
    print(json.dumps(model.analyze(tokens), indent=4, sort_keys=True))
    print(model.tag(tokens))

    # saving the model
    model.save()

# annotate a list of texts, provides results in a list of offset mentions 
def annotate(texts):
    embed_size, embedding_vector = make_embeddings_simple("/mnt/data/wikipedia/embeddings/crawl-300d-2M.vec")
    annotations = []

    # load model
    model = sequenceLabelling.Sequence('insult', embeddings=embedding_vector)
    model.load()
    for text in texts:
        tokens = tokenizeAndFilter(text)
        result = model.analyze(tokens)
        print(json.dumps(result, indent=4, sort_keys=True))
        if result["entities"] is not None:
            entities = result["entities"]
            annotations.append(entities)

    return annotations


if __name__ == "__main__":
    train()
    someTexts = ['This is a gentle test.', 'This is a fucking test!', 'This is a test and I know where you leave.']
    annotate(someTexts)
