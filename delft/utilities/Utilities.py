# some convenient methods for all models
import pandas as pd
import regex as re
import numpy as np
# seed is fixed for reproducibility
from numpy.random import seed
seed(7)
import os.path
import shutil
import requests
from urllib.parse import urlparse

from tensorflow.keras.preprocessing import text

from tqdm import tqdm 

import argparse
import truecase

def truncate_batch_values(batch_values: list, max_sequence_length: int) -> list:
    return [
        row[:max_sequence_length]
        for row in batch_values
    ]

# read list of words (one per line), e.g. stopwords, badwords
def read_words(words_file):
    return [line.replace('\n','').lower() for line in open(words_file, 'r')]


# preprocessing used for twitter-trained glove embeddings
def glove_preprocess(text):
    """
    adapted from https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb

    """
    # Different regex parts for smiley faces
    eyes = "[8:=;]"
    nose = "['`\-]?"
    text = re.sub("https?:* ", "<URL>", text)
    text = re.sub("www.* ", "<URL>", text)
    text = re.sub("\[\[User(.*)\|", '<USER>', text)
    text = re.sub("<3", '<HEART>', text)
    text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
    text = re.sub(eyes + nose + "[Dd)]", '<SMILE>', text)
    text = re.sub("[(d]" + nose + eyes, '<SMILE>', text)
    text = re.sub(eyes + nose + "p", '<LOLFACE>', text)
    text = re.sub(eyes + nose + "\(", '<SADFACE>', text)
    text = re.sub("\)" + nose + eyes, '<SADFACE>', text)
    text = re.sub(eyes + nose + "[/|l*]", '<NEUTRALFACE>', text)
    text = re.sub("/", " / ", text)
    text = re.sub("[-+]?[.\d]*[\d]+[:,.\d]*", "<NUMBER>", text)
    text = re.sub("([!]){2,}", "! <REPEAT>", text)
    text = re.sub("([?]){2,}", "? <REPEAT>", text)
    text = re.sub("([.]){2,}", ". <REPEAT>", text)
    pattern = re.compile(r"(.)\1{2,}")
    text = pattern.sub(r"\1" + " <ELONG>", text)

    return text


# split provided sequence data in two sets given the given ratio between 0 and 1
# for instance ratio at 0.8 will split 80% of the sentence in the first set and 20%
# of the remaining sentence in the second one 
#
def split_data_and_labels(x, y, ratio):
    if (len(x) != len(y)):
        print('error: size of x and y set must be equal, ', len(x), '=/=', len(y))
        return
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for i in range(len(x)):
        if np.random.random_sample() < ratio:
            x1.append(x[i])
            y1.append(y[i])
        else:
            x2.append(x[i])
            y2.append(y[i])
    return np.asarray(x1),np.asarray(y1),np.asarray(x2),np.asarray(y2)    


url_regex = re.compile(r"https?:\/\/[a-zA-Z0-9_\-\.]+(?:com|org|fr|de|uk|se|net|edu|gov|int|mil|biz|info|br|ca|cn|in|jp|ru|au|us|ch|it|nl|no|es|pl|ir|cz|kr|co|gr|za|tw|hu|vn|be|mx|at|tr|dk|me|ar|fi|nz)\/?\b")


# produce some statistics
def stats(x_train=None, y_train=None, x_valid=None, y_valid=None, x_eval=None, y_eval=None):
    charset = []
    nb_total_sequences = 0
    nb_total_tokens = 0
    if x_train is not None:
        print(len(x_train), 'train sequences')
        nb_total_sequences += len(x_train)
        nb_tokens = 0
        for sentence in x_train:
            nb_tokens += len(sentence)
            for token in sentence:
                for character in token:
                    if not character in charset:
                        charset.append(character)
        print("\t","nb. tokens", nb_tokens)
        nb_total_tokens += nb_tokens
    if y_train is not None:
        nb_entities = 0
        for labels in y_train:
            for label in labels:
                if label != 'O':
                    nb_entities += 1
        print("\t","with nb. entities", nb_entities)
    if x_valid is not None:
        print(len(x_valid), 'validation sequences')
        nb_total_sequences += len(x_valid)
        nb_tokens = 0
        for sentence in x_valid:
            nb_tokens += len(sentence)
            for token in sentence:
                for character in token:
                    if not character in charset:
                        charset.append(character)
        print("\t","nb. tokens", nb_tokens)
        nb_total_tokens += nb_tokens
    if y_valid is not None:
        nb_entities = 0
        for labels in y_valid:
            for label in labels:
                if label != 'O':
                    nb_entities += 1
        print("\t","with nb. entities", nb_entities)
    if x_eval is not None:
        print(len(x_eval), 'evaluation sequences')
        nb_total_sequences += len(x_eval)
        nb_tokens = 0
        for sentence in x_eval:
            nb_tokens += len(sentence)
            for token in sentence:
                for character in token:
                    if not character in charset:
                        charset.append(character)
        print("\t","nb. tokens", nb_tokens)
        nb_total_tokens += nb_tokens
    if y_eval is not None:
        nb_entities = 0
        for labels in y_eval:
            for label in labels:
                if label != 'O':
                    nb_entities += 1
        print("\t","with nb. entities", nb_entities)

    print("\n")
    print(nb_total_sequences, "total sequences")
    print(nb_total_tokens, "total tokens\n")

    print("total distinct characters:", len(charset), "\n")
    #print(charset)


# generate the list of out of vocabulary words present in the Toxic dataset 
# with respect to 3 embeddings: fastText, Gloves and word2vec
def generateOOVEmbeddings():
    # read the (DL cleaned) dataset and build the vocabulary
    print('loading dataframes...')
    train_df = pd.read_csv('../data/training/train2.cleaned.dl.csv')
    test_df = pd.read_csv('../data/eval/test2.cleaned.dl.csv')

    # ps: forget memory and runtime, it's python here :D
    list_sentences_train = train_df["comment_text"].values
    list_sentences_test = test_df["comment_text"].values
    list_sentences_all = np.concatenate([list_sentences_train, list_sentences_test])

    tokenizer = text.Tokenizer(num_words=400000)
    tokenizer.fit_on_texts(list(list_sentences_all))
    print('word_index size:', len(tokenizer.word_index), 'words')
    word_index = tokenizer.word_index

    # load fastText - only the words
    print('loading fastText embeddings...')
    voc = set()
    f = open('/mnt/data/wikipedia/embeddings/crawl-300d-2M.vec')
    begin = True
    for line in f:
        if begin:
            begin = False
        else: 
            values = line.split()
            word = ' '.join(values[:-300])
            voc.add(word)
    f.close()
    print('fastText embeddings:', len(voc), 'words')

    oov = []
    for tokenStr in word_index:
        if not tokenStr in voc:
            oov.append(tokenStr)

    print('fastText embeddings:', len(oov), 'out-of-vocabulary')

    with open("../data/training/oov-fastText.txt", "w") as oovFile:
        for w in oov:
            oovFile.write(w)
            oovFile.write('\n')
    oovFile.close()

    # load gloves - only the words
    print('loading gloves embeddings...')
    voc = set()
    f = open('/mnt/data/wikipedia/embeddings/glove.840B.300d.txt')
    for line in f:
        values = line.split()
        word = ' '.join(values[:-300])
        voc.add(word)
    f.close()
    print('gloves embeddings:', len(voc), 'words')

    oov = []
    for tokenStr in word_index:
        if not tokenStr in voc:
            oov.append(tokenStr)

    print('gloves embeddings:', len(oov), 'out-of-vocabulary')

    with open("../data/training/oov-gloves.txt", "w") as oovFile:
        for w in oov:
            oovFile.write(w)
            oovFile.write('\n')
    oovFile.close()

    # load word2vec - only the words
    print('loading word2vec embeddings...')
    voc = set()
    f = open('/mnt/data/wikipedia/embeddings/GoogleNews-vectors-negative300.vec')
    begin = True
    for line in f:
        if begin:
            begin = False
        else: 
            values = line.split()
            word = ' '.join(values[:-300])
            voc.add(word)
    f.close()
    print('word2vec embeddings:', len(voc), 'words')

    oov = []
    for tokenStr in word_index:
        if not tokenStr in voc:
            oov.append(tokenStr)

    print('word2vec embeddings:', len(oov), 'out-of-vocabulary')

    with open("../data/training/oov-w2v.txt", "w") as oovFile:
        for w in oov:    
            oovFile.write(w)
            oovFile.write('\n')
    oovFile.close()

     # load numberbatch - only the words
    print('loading numberbatch embeddings...')
    voc = set()
    f = open('/mnt/data/wikipedia/embeddings/numberbatch-en-17.06.txt')
    begin = True
    for line in f:
        if begin:
            begin = False
        else: 
            values = line.split()
            word = ' '.join(values[:-300])
            voc.add(word)
    f.close()
    print('numberbatch embeddings:', len(voc), 'words')

    oov = []
    for tokenStr in word_index:
        if not tokenStr in voc:
            oov.append(tokenStr)

    print('numberbatch embeddings:', len(oov), 'out-of-vocabulary')

    with open("../data/training/oov-numberbatch.txt", "w") as oovFile:
        for w in oov:    
            oovFile.write(w)
            oovFile.write('\n')
    oovFile.close()


def ontonotes_conll2012_names(pathin, pathout):
    # generate the list of files having a .name extension in the complete ontonotes corpus
    fileout = open(os.path.join(pathout, "names.list"),'w+')

    for subdir, dirs, files in os.walk(pathin):
        for file in files:
            if file.endswith('.name'):
                ind = subdir.find("data/english/")
                if (ind == -1):
                    print("path to ontonotes files appears invalid")
                subsubdir = subdir[ind:]
                fileout.write(os.path.join(subsubdir, file.replace(".name","")))
                fileout.write("\n")
    fileout.close()


def convert_conll2012_to_iob2(pathin, pathout):
    """
    This method will post-process the assembled Ontonotes CoNLL-2012 data for NER. 
    It will take an input like:
      bc/cctv/00/cctv_0001   0    5         Japanese    JJ             *           -    -      -   Speaker#1    (NORP)           *        *            *        *     -
    and transform it into a simple and readable:
      Japanese  B-NORP
    taking into account the sequence markers and an expected IOB2 scheme.
    """
    if pathin == pathout:
        print("input and ouput path must be different:", pathin, pathout)
        return

    names_doc_ids = []
    with open(os.path.join("data", "sequenceLabelling", "CoNLL-2012-NER", "names.list"),'r') as f:
        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                continue
            names_doc_ids.append(line)
    print("number of documents with name notation:", len(names_doc_ids))

    nb_files = 0
     # first pass to get number of files - test files for CoNLL-2012 are under conll-2012-test/, not test/
     # we ignore files not having .names extension in the original ontonotes realease 
    for subdir, dirs, files in os.walk(pathin):
        for file in files:
            if '/english/' in subdir and (file.endswith('gold_conll')) and not '/pt/' in subdir and not '/test/' in subdir:
                ind = subdir.find("data/english/")
                if (ind == -1):
                    print("path to ontonotes files appears invalid")
                subsubdir = os.path.join(subdir[ind:], file.replace(".gold_conll", ""))
                if subsubdir in names_doc_ids:
                    nb_files += 1
    nb_total_files = nb_files
    print(nb_total_files, 'total files to convert')

    # load restricted set of ids for the CoNLL-2012 dataset
    train_doc_ids = []
    dev_doc_ids = []
    test_doc_ids = []

    with open(os.path.join("data", "sequenceLabelling", "CoNLL-2012-NER", "english-ontonotes-5.0-train-document-ids.txt"),'r') as f:
        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                continue
            train_doc_ids.append(line)
    print("number of train documents:", len(train_doc_ids))

    with open(os.path.join("data", "sequenceLabelling", "CoNLL-2012-NER", "english-ontonotes-5.0-development-document-ids.txt"),'r') as f:
        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                continue
            dev_doc_ids.append(line)
    print("number of development documents:", len(dev_doc_ids))

    with open(os.path.join("data", "sequenceLabelling", "CoNLL-2012-NER", "english-ontonotes-5.0-conll-2012-test-document-ids.txt"),'r') as f:
        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                continue
            test_doc_ids.append(line)
    print("number of test documents:", len(test_doc_ids))

    train_out = open(os.path.join(pathout, "eng.train"),'w+', encoding="UTF-8")
    dev_out = open(os.path.join(pathout, "eng.dev"),'w+', encoding="UTF-8")
    test_out = open(os.path.join(pathout, "eng.test"),'w+', encoding="UTF-8")

    nb_files = 0
    pbar = tqdm(total=nb_total_files)
    for subdir, dirs, files in os.walk(pathin):
        for file in files:
            # pt subdirectory corresponds to the old and new testaments, it does not contain NER annotation, so it is traditionally ignored
            #if '/english/' in subdir and (file.endswith('gold_conll') or ('/test/' in subdir and file.endswith('gold_parse_conll'))) and not '/pt/' in subdir:
            if '/english/' in subdir and (file.endswith('gold_conll')) and not '/pt/' in subdir and not '/test/' in subdir:

                ind = subdir.find("data/english/")
                if (ind == -1):
                    print("path to ontonotes files appears invalid")
                subsubdir = os.path.join(subdir[ind:], file.replace(".gold_conll", ""))

                if not subsubdir in names_doc_ids:
                    continue

                pbar.update(1)

                f2 = None
                if '/train/' in subdir and subsubdir in train_doc_ids:
                    f2 = train_out
                elif '/development/' in subdir and subsubdir in dev_doc_ids:
                    f2 = dev_out
                elif '/conll-2012-test/' in subdir and subsubdir in test_doc_ids:
                    f2 = test_out

                if f2 is None:
                    continue

                with open(os.path.join(subdir, file),'r', encoding="UTF-8") as f1:
                    previous_tag = None
                    for line in f1:
                        line_ = line.rstrip()
                        line_ = ' '.join(line_.split())
                        if len(line_) == 0:
                            f2.write("\n")
                            previous_tag = None
                        elif line_.startswith('#begin document'):
                            f2.write(line_+"\n\n")
                            previous_tag = None
                        elif line_.startswith('#end document'):
                            #f2.write("\n")
                            previous_tag = None
                        else:
                            pieces = line_.split(' ')
                            if len(pieces) < 11:
                                print(os.path.join(subdir, file), "-> unexpected number of fiels for line (", len(pieces), "):", line_)
                                previous_tag = None
                            word = pieces[3]
                            # some punctuation are prefixed by / (e.g. /. or /? for dialogue turn apparently)
                            if word.startswith("/") and len(word) > 1:
                                word = word[1:]
                            # in dialogue texts, interjections are maked with a prefix %, e.g. %uh, %eh, we remove this prefix
                            if word.startswith("%") and len(word) > 1:
                                word = word[1:]
                            # there are '='' prefixes to some words, although I don't know what it is used for, we remove it
                            if word.startswith("=") and len(word) > 1:
                                word = word[1:]
                            # we have some markers like -LRB- left bracket, -RRB- right bracket
                            if word == '-LRB-':
                                word = '('
                            if word == '-RRB-':
                                word = ')'
                            # some tokens are identifier in the form 165.00_177.54_B:, 114.86_118.28_A:, and so on, always _A or _B as suffix
                            # it's very unclear why it is in the plain text but clearly noise
                            #regex_str = "\d\d\d\.\d\d_\d\d\d\.\d\d_(A|B)"

                            tag = pieces[10]
                            if tag.startswith('('):
                                if tag.endswith(')'):
                                    tag = tag[1:-1]
                                    previous_tag = None
                                else:
                                    tag = tag[1:-1]
                                    previous_tag = tag
                                f2.write(word+"\tB-"+tag+"\n")
                            elif tag == '*' and previous_tag is not None:
                                f2.write(word+"\tI-"+previous_tag+"\n")
                            elif tag == '*)':
                                f2.write(word+"\tI-"+previous_tag+"\n")
                                previous_tag = None
                            else:
                                f2.write(word+"\tO\n")
                                previous_tag = None
    pbar.close()

    train_out.close()
    dev_out.close()
    test_out.close()


def convert_conll2003_to_iob2(filein, fileout):
    """
    This method will post-process the assembled CoNLL-2003 data for NER. 
    It will take an input like:

    and transform it into a simple and readable:
      Japanese  B-NORP
    taking into account the sequence markers and an expected IOB2 scheme.
    """
    with open(filein,'r') as f1:
        with open(fileout,'w') as f2:
            previous_tag = 'O'
            for line in f1:
                line_ = line.rstrip()
                if len(line_) == 0 or line_.startswith('-DOCSTART-'):
                    f2.write(line_+"\n")
                    previous_tag = 'O'
                else:
                    word, pos, phrase, tag = line_.split(' ')
                    if tag == 'O' or tag.startswith('B-'):
                        f2.write(word+"\t"+tag+"\n")
                    else:
                        subtag = tag[2:]
                        if previous_tag.endswith(tag[2:]):
                            f2.write(word+"\t"+tag+"\n")
                        else:
                            f2.write(word+"\tB-"+tag[2:]+"\n")
                    previous_tag = tag


def merge_folders(root_src_dir, root_dst_dir):
    """
    Recursively merge two folders including subfolders. 
    This method is motivated by the limitation of shutil.copytree() which supposes that the 
    destination directory must not exist.
    """
    for src_dir, dirs, files in os.walk(root_src_dir):
        dst_dir = src_dir.replace(root_src_dir, root_dst_dir, 1)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for file_ in files:
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            shutil.copy(src_file, dst_dir)


def truecase_sentence(tokens):
    """
    from https://github.com/ghaddarAbs
    for experimenting with CoNLL-2003 casing
    """
    word_lst = [(w, idx) for idx, w in enumerate(tokens) if all(c.isalpha() for c in w)]
    lst = [w for w, _ in word_lst if re.match(r'\b[A-Z\.\-]+\b', w)]

    if len(lst) and len(lst) == len(word_lst):
        parts = truecase.get_true_case(' '.join(lst)).split()

        # the trucaser have its own tokenization ...
        # skip if the number of word doesn't match
        if len(parts) != len(word_lst): return tokens

        for (w, idx), nw in zip(word_lst, parts):
            tokens[idx] = nw
    return tokens


def download_file(url, path, filename=None):
    """ 
    Download with Python requests which handle well compression
    """
    # check path
    if path is None or not os.path.isdir(path):
        print("Invalid destination directory:", path)
    HEADERS = {"""User-Agent""": """Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:81.0) Gecko/20100101 Firefox/81.0"""}
    result = "fail"
    print("downloading", url) 
    
    if filename is None:
        # we use the actual server file name
        a = urlparse(url)
        filename = os.path.basename(a.path)
    destination = os.path.join(path, filename)    
    try:
        resp = requests.get(url, stream=True, allow_redirects=True, headers=HEADERS)
        total_length = resp.headers.get('content-length')

        if total_length is None and resp.status_code == 200: 
            # no content length header available, can't have a progress bar :(
            with open(destination, 'wb') as f_out:
                f_out.write(resp.content)
        elif resp.status_code == 200:
            total = int(total_length)
            with open(destination, 'wb') as f_out, tqdm(
                desc=destination,
                total=total,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in resp.iter_content(chunk_size=1024):
                    size = f_out.write(data)
                    bar.update(size)
            result = "success"
    except Exception:
        print("Download failed for {0} with requests".format(url))
    if result == "success":
        return destination
    else:
        return None

def len_until_first_pad(tokens, pad):
    i = len(tokens)-1
    while i>=0:
        if tokens[i] != pad:
            return i+1
        i -= 1
    return 0

def len_until_first_pad_old(tokens, pad):
    for i in range(len(tokens)):
        if tokens[i] == pad:
            return i
    return len(tokens)


if __name__ == "__main__":
    # usage example - for CoNLL-2003, indicate the eng.* file to be converted:
    # > python3 utilities/Utilities.py --dataset-type conll2003 --data-path /home/lopez/resources/CoNLL-2003/eng.train --output-path /home/lopez/resources/CoNLL-2003/iob2/eng.train 
    # for CoNLL-2012, indicate the root directory of the ontonotes data (in CoNLL-2012 format) to be converted:
    # > python3 utilities/Utilities.py --dataset-type conll2012 --data-path /home/lopez/resources/ontonotes/conll-2012/ --output-path /home/lopez/resources/ontonotes/conll-2012/iob2/

    # get the argument
    parser = argparse.ArgumentParser(
        description = "Named Entity Recognizer dataset converter to OIB2 tagging scheme")

    #parser.add_argument("action")
    parser.add_argument("--dataset-type",default='conll2003', help="dataset to be used for training the model, one of ['conll2003','conll2012']")
    parser.add_argument("--data-path", default=None, help="path to the corpus of documents to process") 
    parser.add_argument("--output-path", default=None, help="path to write the converted dataset") 

    args = parser.parse_args()

    #action = args.action 
    dataset_type = args.dataset_type
    data_path = args.data_path
    output_path = args.output_path

    if dataset_type == 'conll2003':
        convert_conll2003_to_iob2(data_path, output_path)
    elif dataset_type == 'conll2012':    
        convert_conll2012_to_iob2(data_path, output_path)
    elif dataset_type == 'ontonotes':    
        ontonotes_conll2012_names(data_path, output_path)
