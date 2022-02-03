import os

from transformers import AutoTokenizer, TFAutoModel, AutoConfig

TRANSFORMER_CONFIG_FILE_NAME = 'transformer-config.json'

class Transformer(object):
    def __init__(self, resource_registry=None):
        self.registry = resource_registry
        self.tokenizer = None
        self.model = None
        self.local_path = None

    def load(self, name, max_sequence_length, add_special_tokens=True, add_prefix_space=True):
        # TODO: Add loading from local

        self.name = name
        if 'transformers' not in self.registry:
            load_name = name
        else:
            filtered_resources = list(
                filter(lambda x: 'name' in x and x['name'] == self.name, self.registry['transformers']))
            if len(filtered_resources) > 0:
                transformer_configuration = list(filtered_resources)[0]
                self.local_path = transformer_configuration['model_dir']
                load_name = self.local_path
            else:
                load_name = name

        self.tokenizer = AutoTokenizer.from_pretrained(load_name,
                                                           add_special_tokens=add_special_tokens,
                                                           max_length=max_sequence_length,
                                                           add_prefix_space=add_prefix_space)

    def get_tokenizer(self):
        return self.tokenizer

    def get_layers(self, transformer_model_name, load_pretrained_weights=True, local_path=None):
        if load_pretrained_weights:
            if local_path is None:
                transformer_model = TFAutoModel.from_pretrained(transformer_model_name, from_pt=True)
            else:
                transformer_model = TFAutoModel.from_pretrained(local_path, from_pt=True)
            self.bert_config = transformer_model.config
        else:
            # load config in JSON format
            if local_path is None:
                self.bert_config = AutoConfig.from_pretrained(transformer_model_name)
            else:
                config_path = os.path.join(".", local_path, TRANSFORMER_CONFIG_FILE_NAME)
                self.bert_config = AutoConfig.from_pretrained(config_path)
            transformer_model = TFAutoModel.from_config(self.bert_config)
        return transformer_model

    def get_model_local_path(self):
        return self.local_path
