import argparse
import json
import time

from sklearn.model_selection import train_test_split

from delft.sequenceLabelling import Sequence
from delft.sequenceLabelling.models import *
from delft.sequenceLabelling.reader import load_data_and_labels_crf_file

import keras.backend as K

"""
grobid-affiliation-address/
grobid-citation/
grobid-citation-with_ELMo/
grobid-date/
grobid-date-with_ELMo/
grobid-figure/
grobid-header/
grobid-name-citation/
grobid-name-header/
grobid-quantities/
grobid-quantities-with_ELMo/
grobid-reference-segmenter/
grobid-software/
grobid-table/
grobid-units/
grobid-units-with_ELMo/
grobid-values/
grobid-values-with_ELMo/
insult/
ner-en-conll2003-BidGRU_CRF/
ner-en-conll2003-BidLSTM_CNN/
ner-en-conll2003-BidLSTM_CNN_CRF/
ner-en-conll2003-BidLSTM_CRF/
ner-en-conll2003-BidLSTM_CRF_CASING/
ner-en-conll2003-with_BERT-BidLSTM_CRF/
ner-en-conll2003-with_ELMo-BidGRU_CRF/
ner-en-conll2003-with_ELMo-BidLSTM_CNN/
ner-en-conll2003-with_ELMo-BidLSTM_CNN_CRF/
ner-en-conll2003-with_ELMo-BidLSTM_CRF/
ner-en-conll2003-with_ELMo-BidLSTM_CRF_CASING/
ner-en-conll2012-BidLSTM_CRF/
ner-en-conll2012-with_ELMo-BidLSTM_CRF/
ner-fr-lemonde-BidLSTM_CRF/
ner-fr-lemonde-force-split-BidLSTM_CRF/
ner-fr-lemonde-force-split-with_ELMo-BidLSTM_CRF/
ner-fr-lemonde-with_ELMo-BidLSTM_CRF
"""

ARCHITECTURE = ['BidGRU_CRF', 'BidLSTM_CRF', 'BidLSTM_CNN_CRF', 'BidLSTM_CRF_CASING', 'BidLSTM_CNN']
GROBID_MODEL = ['affiliation-address', 'citation', 'date', 'header', 'figure', 'name-citation', 'name-header', 'software', 
    'quantities', 'reference-segmenter','table', 'units', 'values']
NER_MODELS_EN = ['conll2003', 'conll2012']
NER_MODELS_FR = ['lemonde', 'lemonde-force-split']

DATA_PATH = "data/models/sequenceLabelling/"

def migrate():
    # grobid models
    for grobid_model in GROBID_MODEL:
        model_name = 'grobid-' + grobid_model

        # load the model
        """
        print(os.path.join(DATA_PATH, model_name))
        if os.path.isdir(os.path.join(DATA_PATH, model_name)):
            model = Sequence(model_name)
            model.load()
            model.save()
        """

        # with ELMo
        """
        if os.path.isdir(os.path.join(DATA_PATH, model_name+'-with_ELMo')):
            model = Sequence(model_name+'-with_ELMo')
            model.load()
            model.save()
        """

    # insult model
    """
    model = Sequence('insult')
    model.load()
    model.save()
    """

    for en_model in NER_MODELS_EN:
        for architecture in ARCHITECTURE:
            """
            model_name = 'ner-en-' + en_model
            model_name += '-' + architecture
            if os.path.isdir(os.path.join(DATA_PATH, model_name)):
                model = Sequence(model_name)
                model.load()
                model.save()
            """

            """
            model_name = 'ner-en-' + en_model
            model_name += '-with_ELMo'
            model_name += '-' + architecture
            if os.path.isdir(os.path.join(DATA_PATH, model_name)):
                model = Sequence(model_name)
                model.load()
                model.save()

            """
            """
            model_name = 'ner-en-' + en_model
            model_name += '-with_BERT'
            model_name += '-' + architecture
            if os.path.isdir(os.path.join(DATA_PATH, model_name)):
                model = Sequence(model_name)
                model.load()
                model.save()
            """

    for fr_model in NER_MODELS_FR:
        for architecture in ARCHITECTURE:
            
            model_name = 'ner-fr-' + fr_model
            model_name += '-' + architecture
            print(os.path.join(DATA_PATH, model_name))
            if os.path.isdir(os.path.join(DATA_PATH, model_name)):
                model = Sequence(model_name)
                model.load()
                model.save()
            

            model_name = 'ner-fr-' + fr_model
            model_name += '-with_ELMo'
            model_name += '-' + architecture
            if os.path.isdir(os.path.join(DATA_PATH, model_name)):
                model = Sequence(model_name)
                model.load()
                model.save()
            
            
if __name__ == "__main__":
    migrate()