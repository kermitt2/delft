import os

from transformers import AutoTokenizer, TFAutoModel, AutoConfig

TRANSFORMER_CONFIG_FILE_NAME = 'transformer-config.json'

class Transformer(object):
    def __init__(self, resource_registry=None):
        self.registry = resource_registry
        self.tokenizer = None
        self.model = None
        self.local_path = None

        self.config_file = None
        self.vocab_file = None
        self.weight_file = None

    def load_tokenizer(self, name, max_sequence_length, add_special_tokens=True, add_prefix_space=True):
        # TODO: Add loading from local original

        self.name = name
        if 'transformers' not in self.registry:
            load_name = name
        else:
            filtered_resources = list(
                filter(lambda x: 'name' in x and x['name'] == self.name, self.registry['transformers']))
            if len(filtered_resources) > 0:
                transformer_configuration = list(filtered_resources)[0]
                if 'model_dir' in transformer_configuration:
                    self.local_path = transformer_configuration['model_dir']
                    self.tokenizer = AutoTokenizer.from_pretrained(self.local_path,
                                                                   add_special_tokens=add_special_tokens,
                                                                   max_length=max_sequence_length,
                                                                   add_prefix_space=add_prefix_space)
                else:
                    # self.config_file = None
                    # self.weight_file = None
                    self.vocab_file = None
                    # if "path-config" in transformer_configuration and os.path.isfile(transformer_configuration["path-config"]):
                    #     self.config_file = transformer_configuration["path-config"]
                    # if "path-weights" in transformer_configuration and os.path.isfile(transformer_configuration["path-weights"]+".data-00000-of-00001"):
                    #     self.weight_file = transformer_configuration["path-weights"]
                    if "path-vocab" in transformer_configuration and os.path.isfile(transformer_configuration["path-vocab"]):
                        self.vocab_file = transformer_configuration["path-vocab"]
                        self.tokenizer = AutoTokenizer.from_pretrained(self.vocab_file)

            else:
                self.tokenizer = AutoTokenizer.from_pretrained(name,
                                                               add_special_tokens=add_special_tokens,
                                                               max_length=max_sequence_length,
                                                               add_prefix_space=add_prefix_space)



    def get_tokenizer(self):
        return self.tokenizer

    def instantiate_layer(self, transformer_model_name, load_pretrained_weights=True, local_path=None):
        if load_pretrained_weights:
            return self.instantiate_layer_with_weights(transformer_model_name, local_path)
        else:
            return self.instantiate_layer_empty(transformer_model_name, local_path)

    def instantiate_layer_empty(self, transformer_model_name, local_path=None):
        if local_path is None:
            transformer_model = TFAutoModel.from_pretrained(transformer_model_name, from_pt=True)
        else:
            transformer_model = TFAutoModel.from_pretrained(local_path, from_pt=True)

        self.bert_config = transformer_model.config

        transformer_model = TFAutoModel.from_config(self.bert_config)
        return transformer_model

    def instantiate_layer_with_weights(self, transformer_model_name, local_path=None):
        # load config in JSON format
        if local_path is None:
            self.bert_config = AutoConfig.from_pretrained(transformer_model_name)
        else:
            config_path = os.path.join(".", local_path, TRANSFORMER_CONFIG_FILE_NAME)
            self.bert_config = AutoConfig.from_pretrained(config_path)

    def get_model_local_path(self):
        return self.local_path
