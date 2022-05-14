import os
from typing import Union, Iterable

from transformers import AutoTokenizer, TFAutoModel, AutoConfig, BertTokenizer, TFBertModel

TRANSFORMER_CONFIG_FILE_NAME = 'transformer-config.json'
DEFAULT_TRANSFORMER_TOKENIZER_DIR = "transformer-tokenizer"

LOADING_METHOD_LOCAL_MODEL_DIR = "local_model_dir"
LOADING_METHOD_HUGGINGFACE_NAME = "huggingface"
LOADING_METHOD_PLAIN_MODEL = "plain_model"
LOADING_METHOD_DELFT_MODEL = "delft_model"


class Transformer(object):
    """
    This class provides a wrapper around a transformer model (pre-trained or fine-tuned)
    
    This class makes possible to load a transformer config, tokenizer and weights using different approaches, 
    prioritizing the most "local" method over an external access to the HuggingFace Hub:

     1. loading the transformer saved locally as part of an existing full delft model (the configuration file 
        name will be different)
     2. via a local directory
     3. by specifying the weight, config and vocabulary files separately (this is likely the scenario where 
        the user try to load a model from the downloaded bert/scibertt model from github
     4. via the HuggingFace transformer name and HuggingFace Hub, resulting in several online requests to this 
        Hub, which could fail if the service is overloaded
    """

    def __init__(self, name: str, resource_registry: dict = None, delft_local_path: str = None):

        self.bert_preprocessor = None
        self.transformer_config = None
        self.loading_method = None
        self.model = None

        # In case the model is loaded from a local directory
        self.local_dir_path = None

        # In case the weights, config and vocab are specified separately (model vanilla)
        self.local_weights_file = None
        self.local_config_file = None
        self.local_vocab_file = None

        self.name = name

        if delft_local_path:
            self.loading_method = LOADING_METHOD_DELFT_MODEL
            self.local_dir_path = delft_local_path

        self.tokenizer = None

        if resource_registry:
            self.configure_from_registry(resource_registry)

    def configure_from_registry(self, resource_registry) -> None:
        """
        Fetch transformer information from the registry and infer the loading method:
            1. if no configuration is provided is using HuggingFace with the provided name
            2. if only the directory is provided it will load the model from that directory
            3. if the weights, config and vocab are provided (as in the vanilla models) then 
               it will load them as BertTokenizer and BertModel
        """

        if self.loading_method == LOADING_METHOD_DELFT_MODEL:
            return

        if 'transformers' in resource_registry:
            filtered_resources = list(
                filter(lambda x: 'name' in x and x['name'] == self.name, resource_registry['transformers']))
            if len(filtered_resources) > 0:
                transformer_configuration = list(filtered_resources)[0]
                if 'model_dir' in transformer_configuration:
                    self.local_dir_path = transformer_configuration['model_dir']
                    self.loading_method = LOADING_METHOD_LOCAL_MODEL_DIR
                else:
                    self.loading_method = LOADING_METHOD_PLAIN_MODEL
                    if "path-config" in transformer_configuration and os.path.isfile(
                            transformer_configuration["path-config"]):
                        self.local_config_file = transformer_configuration["path-config"]
                    else:
                        print("Missing path-config or not a file.")

                    if "path-weights" in transformer_configuration and os.path.isfile(
                            transformer_configuration["path-weights"]) or os.path.isfile(
                        transformer_configuration["path-weights"] + ".data-00000-of-00001"):
                        self.local_weights_file = transformer_configuration["path-weights"]
                    else:
                        print("Missing weights-config or not a file.")

                    if "path-vocab" in transformer_configuration and os.path.isfile(
                            transformer_configuration["path-vocab"]):
                        self.local_vocab_file = transformer_configuration["path-vocab"]
                    else:
                        print("Missing vocab-file or not a file.")
            else:
                self.loading_method = LOADING_METHOD_HUGGINGFACE_NAME
                #print("No configuration for", self.name, "Loading from Hugging face.")
        else:
            self.loading_method = LOADING_METHOD_HUGGINGFACE_NAME
            #print("No configuration for", self.name, "Loading from Hugging face.")

    def init_preprocessor(self, max_sequence_length: int,
                       add_special_tokens: bool = True,
                       add_prefix_space: bool = True):
        """
        Load the tokenizer according to the provided information, in case of missing configuration,
        it will try to use huggingface as fallback solution.
        """
        if self.loading_method == LOADING_METHOD_HUGGINGFACE_NAME:
            self.tokenizer = AutoTokenizer.from_pretrained(self.name,
                                                           add_special_tokens=add_special_tokens,
                                                           max_length=max_sequence_length,
                                                           add_prefix_space=add_prefix_space)

        elif self.loading_method == LOADING_METHOD_LOCAL_MODEL_DIR:
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir_path,
                                                           add_special_tokens=add_special_tokens,
                                                           max_length=max_sequence_length,
                                                           add_prefix_space=add_prefix_space)
        elif self.loading_method == LOADING_METHOD_PLAIN_MODEL:
            self.tokenizer = BertTokenizer.from_pretrained(self.local_vocab_file)

        elif self.loading_method == LOADING_METHOD_DELFT_MODEL:
            config_path = os.path.join(".", self.local_dir_path, TRANSFORMER_CONFIG_FILE_NAME)
            self.transformer_config = AutoConfig.from_pretrained(config_path)
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.local_dir_path, DEFAULT_TRANSFORMER_TOKENIZER_DIR), config=self.transformer_config)

    def save_tokenizer(self, output_directory):
        self.tokenizer.save_pretrained(output_directory)

    def instantiate_layer(self, load_pretrained_weights=True) -> Union[object, TFAutoModel, TFBertModel]:
        """
        Instanciate a transformer to be loaded in a Keras layer using the availability method of the pre-trained transformer.
        """
        if self.loading_method == LOADING_METHOD_HUGGINGFACE_NAME:
            if load_pretrained_weights:
                transformer_model = TFAutoModel.from_pretrained(self.name, from_pt=True)
                self.transformer_config = transformer_model.config
                return transformer_model
            else:
                config_path = os.path.join(".", self.local_dir_path, TRANSFORMER_CONFIG_FILE_NAME)
                self.transformer_config = AutoConfig.from_pretrained(config_path)
                return TFAutoModel.from_config(self.transformer_config)

        elif self.loading_method == LOADING_METHOD_LOCAL_MODEL_DIR:
            if load_pretrained_weights:
                transformer_model = TFAutoModel.from_pretrained(self.local_dir_path, from_pt=True)
                self.transformer_config = transformer_model.config
                return transformer_model
            else:
                config_path = os.path.join(".", self.local_dir_path, TRANSFORMER_CONFIG_FILE_NAME)
                self.transformer_config = AutoConfig.from_pretrained(config_path)
                #self.transformer_config = AutoConfig.from_pretrained(self.local_dir_path)
                return TFAutoModel.from_config(self.transformer_config)

        elif self.loading_method == LOADING_METHOD_PLAIN_MODEL:
            if load_pretrained_weights:
                self.transformer_config = AutoConfig.from_pretrained(self.local_config_file)
                # transformer_model = TFBertModel.from_pretrained(self.local_weight_file, from_tf=True)
                raise NotImplementedError(
                    "The load of TF weights from huggingface automodel classes is not yet implemented. \
                    Please use load from Hugging Face Hub or from directory for the initial loading of the transformers weights.")
            else:
                config_path = os.path.join(".", self.local_dir_path, TRANSFORMER_CONFIG_FILE_NAME)
                self.transformer_config = AutoConfig.from_pretrained(config_path)
                return TFBertModel.from_config(self.transformer_config)

        else:
            # TODO: revise this
            if load_pretrained_weights:
                transformer_model = TFAutoModel.from_pretrained(self.local_dir_path, from_pt=True)
                self.transformer_config = transformer_model.config
                return transformer_model
            else:
                config_path = os.path.join(".", self.local_dir_path, TRANSFORMER_CONFIG_FILE_NAME)
                self.transformer_config = AutoConfig.from_pretrained(config_path)
                return TFAutoModel.from_config(self.transformer_config)
