import json
import os
import numpy as np

import bert
from bert import BertModelLayer
from bert.tokenization.bert_tokenization import FullTokenizer

class BERT_layer():
    """
    BERT layer to be used in a Keras model.

    For reference:
    --
    @article{devlin2018bert,
      title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
      author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
      journal={arXiv preprint arXiv:1810.04805},
      year={2018}
    }
    """

    def __init__(self, model_name):

        # we get the BERT pretrained files from the embeddings registry
        description = _get_description(model_name)

        if description == None:
            raise Exception('no pre-trained model description found for ' + self.model_name)

        # note: postpone the instanciation if not available, it normally means that 
        # we load a fine-tuned model and we don't need to look at the original
        # pre-trained resources (this is mandatory for the vocabulary when predicting)
        self.config_file = None
        self.weight_file = None
        self.vocab_file = None
        if description != None:
            if "path-config" in description:
                if os.path.isfile(description["path-config"]):
                    self.config_file = description["path-config"]
                else:
                    print("check embeddings registry: invalid file path for", description["path-config"])
            else:
                print("config file for", model_name, "not specified in embeddings registry")
            if "path-weights" in description and os.path.isfile(description["path-weights"]+".data-00000-of-00001"):
                self.weight_file = description["path-weights"] 
            if "path-vocab" in description and os.path.isfile(description["path-vocab"]):
                self.vocab_file = description["path-vocab"]

        # defaulting to fine-tuned model if available
        '''
        if self.config_file == None:
            self.config_file = os.path.join(self.model_dir, 'bert_config.json')
        if self.weight_file == None:
            self.weight_file = os.path.join(self.model_dir, 'model.ckpt') 
        if self.vocab_file == None: 
            self.vocab_file = os.path.join(self.model_dir, 'vocab.txt')
        '''

        self.tokenizer = FullTokenizer(vocab_file=self.vocab_file, do_lower_case=False)
        self.bert_config = modeling.BertConfig.from_json_file(self.config_file)

        bert_params = bert.params_from_pretrained_ckpt(self.model_dir)
        self.l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")

    def load_weights(self):
        # load the pre-trained model weights
        bert.load_bert_weights(self.l_bert, self.weight_file)  

    def get_layer(self):
        return self.l_bert

# TBD: use a common one 
def _get_description(name, path="./embedding-registry.json"):
    registry_json = open(path).read()
    registry = json.loads(registry_json)
    for emb in registry["transformers"]:
            if emb["name"] == name:
                return emb
    return None
