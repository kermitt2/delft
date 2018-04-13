import os
import json
from utilities.Embeddings import make_embeddings_simple
import sequenceLabelling
from sequenceLabelling.tokenizer import tokenizeAndFilter
from sequenceLabelling.reader import load_data_and_labels_xml_file, load_data_and_labels_conll

max_features = 100000
embed_size = 300

def train(): 
    root = os.path.join(os.path.dirname(__file__), '../data/sequence/')

    train_path = os.path.join(root, 'corrected.xml')
    valid_path = os.path.join(root, 'valid.xml')

    print('Loading data...')
    x_train, y_train = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.train')
    x_valid, y_valid = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testa')
    x_test, y_test = load_data_and_labels_conll('data/sequenceLabelling/CoNLL-2003/eng.testb')
    print(len(x_train), 'train sequences')
    print(len(x_valid), 'validation sequences')
    print(len(x_test), 'evaluation sequences')

    embed_size, embedding_vector = make_embeddings_simple("/mnt/data/wikipedia/embeddings/crawl-300d-2M.vec")

    model = sequenceLabelling.Sequence('ner', max_epoch=25, embeddings=embedding_vector, word_emb_size=embed_size)
    model.train(x_train, y_train, x_valid, y_valid)

    tokens = tokenizeAndFilter('The University of California has found that 40 percent of its students suffer food insecurity. At four state universities in Illinois, that number is 35 percent.')
    print(json.dumps(model.analyze(tokens), indent=4, sort_keys=True))
    print(model.tag(tokens))

    tokens = tokenizeAndFilter('President Obama is not speaking anymore from the White House.')
    print(json.dumps(model.analyze(tokens), indent=4, sort_keys=True))
    print(model.tag(tokens))

    model.eval(x_test, y_test)

    # saving the model
    model.save()

# annotate a list of texts, provides results in a list of offset mentions 
def annotate(texts):
    embed_size, embedding_vector = make_embeddings_simple("/mnt/data/wikipedia/embeddings/crawl-300d-2M.vec")
    annotations = []

    # load model
    model = sequenceLabelling.Sequence('ner', embeddings=embedding_vector)
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
    someTexts = ['']
    annotate(someTexts)