import json
import os
import numpy as np

from .bert import BertModelLayer, params_from_pretrained_ckpt, load_bert_weights

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
            raise Exception('no pre-trained model description found for ' + model_name)

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
            if "model_dir" in description:
                self.model_dir = description["model_dir"]

        if self.config_file != None:
            with open(self.config_file) as json_file:
                bert_params_dict = json.load(json_file)
                #bert_params_dict["return_pooler_output"] = True
                print(bert_params_dict)
                bert_params = _config_to_params(bert_params_dict)
        else:
            bert_params = params_from_pretrained_ckpt(self.model_dir)
        self.l_bert = BertModelLayer.from_params(bert_params, name="bert")

    def load_weights(self):
        # load the pre-trained model weights
        load_bert_weights(self.l_bert, self.weight_file)  

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

def _get_vocab_file_path(name, path="./embedding-registry.json"):
    print(name)
    registry_json = open(path).read()
    registry = json.loads(registry_json)
    for emb in registry["transformers"]:
        if emb["name"] == name:
            print(emb)
            if "path-vocab" in emb and os.path.isfile(emb["path-vocab"]):
                return emb["path-vocab"]
    return None

def _config_to_params(bc):
    """
    because map_stock_config_to_params is not working with a dict
    """
    bert_params = BertModelLayer.Params(
        num_layers=bc.get("num_hidden_layers"),
        num_heads=bc.get("num_attention_heads"),
        hidden_size=bc.get("hidden_size"),
        hidden_dropout=bc.get("hidden_dropout_prob"),
        attention_dropout=bc.get("attention_probs_dropout_prob"),

        intermediate_size=bc.get("intermediate_size"),
        intermediate_activation=bc.get("hidden_act"),

        vocab_size=bc.get("vocab_size"),
        use_token_type=True,
        use_position_embeddings=True,
        token_type_vocab_size=bc.get("type_vocab_size"),
        max_position_embeddings=bc.get("max_position_embeddings"),

        embedding_size=bc.get("embedding_size"),
        shared_layer=bc.get("embedding_size") is not None,
    )
    return bert_params   