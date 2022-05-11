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
    This class provides a wrapper around the Transformer.
    With this class is possible to load a transformer tokenizer and layer using different approaches:
     1. via the hugging face name (beware this will result in several requests to huggingface.co, which could fail if the service is overloaded
     2. via a local directory
     3. by specifying the weight, config and vocabulary files separately (this is likely the scenario where the user try to load a model from the downloaded bert/scibertt model from github
     4. loading the transformer as part of the delft model (the configuration file name will be different)
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
            1. if no configuration is provided is using huggingface with the provided name
            2. if only the directory is provided it will load the model from that directory
            3. if the weights, config and vocab are provided (as in the vanilla models) then it will load them as BertTokenizer and BertModel
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
                       add_prefix_space: bool = True,
                       empty_features_vector: Iterable[int] =None,
                       empty_char_vector: Iterable[int] = None) -> BERTPreprocessor:
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

        self.bert_preprocessor = BERTPreprocessor(self.tokenizer, empty_features_vector, empty_char_vector)

        return self.bert_preprocessor

    def save_tokenizer(self, output_directory):
        self.tokenizer.save_pretrained(output_directory)

    def instantiate_layer(self, load_pretrained_weights=True) -> Union[object, TFAutoModel, TFBertModel]:
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



class BERTPreprocessor(object):
    """
    Generic BERT preprocessor for a sequence labelling data set.
    Input are pre-tokenized texts, possibly with features and labels to re-align with the sub-tokenization.
    Rely on transformers library tokenizer
    """

    def __init__(self, tokenizer, empty_features_vector=None, empty_char_vector=None):
        self.tokenizer = tokenizer
        self.empty_features_vector = empty_features_vector
        self.empty_char_vector = empty_char_vector

    def tokenize_and_align_features_and_labels(self, texts, chars, text_features, text_labels, maxlen=512):
        """
        Training/evaluation usage with features: sub-tokenize+convert to ids/mask/segments input texts, realign labels
        and features given new tokens introduced by the wordpiece sub-tokenizer.
        texts is a list of texts already pre-tokenized
        """
        target_ids = []
        target_type_ids = []
        target_attention_mask = []
        input_tokens = []
        target_chars = []

        target_features = None
        if text_features is not None:
            target_features = []

        target_labels = None
        if text_labels is not None:
            target_labels = []

        for i, text in enumerate(texts):

            local_chars = chars[i]

            features = None
            if text_features is not None:
                features = text_features[i]

            label_list = None
            if text_labels is not None:
                label_list = text_labels[i]

            input_ids, token_type_ids, attention_mask, chars_block, feature_blocks, target_tags, tokens = self.convert_single_text(
                text,
                local_chars,
                features,
                label_list,
                maxlen)
            target_ids.append(input_ids)
            target_type_ids.append(token_type_ids)
            target_attention_mask.append(attention_mask)
            input_tokens.append(tokens)
            target_chars.append(chars_block)

            if target_features is not None:
                target_features.append(feature_blocks)

            if target_labels is not None:
                target_labels.append(target_tags)

        return target_ids, target_type_ids, target_attention_mask, target_chars, target_features, target_labels, input_tokens

    def convert_single_text(self, text_tokens, chars_tokens, features_tokens, label_tokens, max_seq_length):
        """
        Converts a single sequence input into a single transformer input format using generic tokenizer
        of the transformers library, align other channel input to the new sub-tokenization
        """
        if label_tokens is None:
            # we create a dummy label list to facilitate
            label_tokens = []
            while len(label_tokens) < len(text_tokens):
                label_tokens.append(0)

        if features_tokens is None:
            # we create a dummy feature list to facilitate
            features_tokens = []
            while len(features_tokens) < len(text_tokens):
                features_tokens.append(self.empty_features_vector)

        if chars_tokens is None:
            # we create a dummy feature list to facilitate
            chars_tokens = []
            while len(chars_tokens) < len(text_tokens):
                chars_tokens.append(self.empty_char_vector)

        # sub-tokenization
        encoded_result = self.tokenizer(text_tokens, add_special_tokens=True, is_split_into_words=True,
                                        max_length=max_seq_length, truncation=True, return_offsets_mapping=True)

        input_ids = encoded_result.input_ids
        offsets = encoded_result.offset_mapping
        if "token_type_ids" in encoded_result:
            token_type_ids = encoded_result.token_type_ids
        else:
            token_type_ids = [0] * len(input_ids)
        attention_mask = encoded_result.attention_mask
        label_ids = []
        chars_blocks = []
        feature_blocks = []

        # trick to support sentence piece tokenizer like GPT2, roBERTa, CamemBERT, etc. which encode prefixed
        # spaces in the tokens (the encoding symbol for this space varies from one model to another)
        new_input_ids = []
        new_attention_mask = []
        new_token_type_ids = []
        new_offsets = []
        for i in range(0, len(input_ids)):
            if len(self.tokenizer.decode(input_ids[i])) != 0:
                # if a decoded token has a length of 0, it is typically a space added for sentence piece/camembert/GPT2
                # which happens to be then sometimes a single token for unknown reason when with is_split_into_words=True
                # we need to skip this but also remove it from attention_mask, token_type_ids and offsets to stay
                # in sync
                new_input_ids.append(input_ids[i])
                new_attention_mask.append(attention_mask[i])
                new_token_type_ids.append(token_type_ids[i])
                new_offsets.append(offsets[i])
        input_ids = new_input_ids
        attention_mask = new_attention_mask
        token_type_ids = new_token_type_ids
        offsets = new_offsets

        word_idx = -1
        for i, offset in enumerate(offsets):
            if offset[0] == 0 and offset[1] == 0:
                # this is a special token
                label_ids.append("<PAD>")
                chars_blocks.append(self.empty_char_vector)
                feature_blocks.append(self.empty_features_vector)
            else:
                if offset[0] == 0:
                    word_idx += 1

                    # new token
                    label_ids.append(label_tokens[word_idx])
                    feature_blocks.append(features_tokens[word_idx])
                    chars_blocks.append(chars_tokens[word_idx])
                else:
                    # propagate the data to the new sub-token or
                    # dummy/empty input for sub-tokens
                    label_ids.append("<PAD>")
                    chars_blocks.append(self.empty_char_vector)
                    # 2 possibilities, either empty features for sub-tokens or repeating the
                    # feature vector of the prefix sub-token
                    # feature_blocks.append(self.empty_features_vector)
                    feature_blocks.append(features_tokens[word_idx])

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(self.tokenizer.pad_token_id)
            token_type_ids.append(self.tokenizer.pad_token_id)
            attention_mask.append(0)
            label_ids.append("<PAD>")
            chars_blocks.append(self.empty_char_vector)
            feature_blocks.append(self.empty_features_vector)

        assert len(input_ids) == max_seq_length
        assert len(token_type_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(chars_blocks) == max_seq_length
        assert len(feature_blocks) == max_seq_length

        return input_ids, token_type_ids, attention_mask, chars_blocks, feature_blocks, label_ids, offsets

    def convert_single_text_bert(self, text_tokens, chars_tokens, features_tokens, label_tokens, max_seq_length):
        """
        Converts a single sequence input into a single BERT input format and align other channel input to this
        new sub-tokenization

        The original BERT implementation works as follow:

        input:
            tokens: [Jim,Henson,was,a,puppeteer]
            labels: [I-PER,I-PER,O,O,O]
        The BERT tokenization will modify sentence and labels as follow:
            tokens: [Jim,Hen,##son,was,a,puppet,##eer]
            labels: [I-PER,I-PER,X,O,O,O,X]

        Here, we don't introduce the X label for added sub-tokens and simply extend the previous label
        and produce:
            tokens: [Jim,Hen,##son,was,a,puppet,##eer]
            labels: [I-PER,I-PER,I-PER,O,O,O,O]

        Notes: input and output labels are text labels, they need to be changed into indices after
        text conversion.
        """

        tokens = []
        # the following is to keep track of additional tokens added by BERT tokenizer,
        # only some of them has a prefix ## that allows to identify them downstream in the process
        tokens_marked = []
        labels = []
        features = []
        chars = []

        if label_tokens is None:
            # we create a dummy label list to facilitate
            label_tokens = []
            while len(label_tokens) < len(text_tokens):
                label_tokens.append(0)

        if features_tokens is None:
            # we create a dummy feature list to facilitate
            features_tokens = []
            while len(features_tokens) < len(text_tokens):
                features_tokens.append(self.empty_features_vector)

        if chars_tokens is None:
            # we create a dummy feature list to facilitate
            chars_tokens = []
            while len(chars_tokens) < len(text_tokens):
                chars_tokens.append(self.empty_char_vector)

        for text_token, label_token, chars_token, features_token in zip(text_tokens, label_tokens, chars_tokens,
                                                                        features_tokens):
            text_sub_tokens = self.tokenizer.tokenize(text_token, add_special_tokens=False)

            # we mark added sub-tokens with the "##" prefix in order to restore token back correctly,
            # otherwise the BERT tokenizer do not mark them all with this prefix
            # (note: to be checked if it's the same with the non-original BERT tokenizer)
            text_sub_tokens_marked = self.tokenizer.tokenize(text_token, add_special_tokens=False)
            for i in range(len(text_sub_tokens_marked)):
                if i == 0:
                    continue
                tok = text_sub_tokens_marked[i]
                if not tok.startswith("##"):
                    text_sub_tokens_marked[i] = "##" + tok

            label_sub_tokens = [label_token] + [label_token] * (len(text_sub_tokens) - 1)
            chars_sub_tokens = [chars_token] + [chars_token] * (len(text_sub_tokens) - 1)
            feature_sub_tokens = [features_token] + [features_token] * (len(text_sub_tokens) - 1)

            tokens.extend(text_sub_tokens)
            tokens_marked.extend(text_sub_tokens_marked)
            labels.extend(label_sub_tokens)
            features.extend(feature_sub_tokens)
            chars.extend(chars_sub_tokens)

        if len(tokens) >= max_seq_length - 2:
            tokens = tokens[0:(max_seq_length - 2)]
            tokens_marked = tokens_marked[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            features = features[0:(max_seq_length - 2)]
            chars = chars[0:(max_seq_length - 2)]

        input_tokens = []
        input_tokens_marked = []
        segment_ids = []
        label_ids = []
        chars_blocks = []
        feature_blocks = []

        # The convention in BERT is:
        # (a) For sequence pairs:
        #   tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #   type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #   tokens:   [CLS] the dog is hairy . [SEP]
        #   type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first sequence
        # or the second sequence.

        input_tokens.append(self.tokenizer.cls_token)
        input_tokens_marked.append(self.tokenizer.cls_token)
        segment_ids.append(0)
        label_ids.append("<PAD>")
        chars_blocks.append(self.empty_char_vector)
        feature_blocks.append(self.empty_features_vector)

        for i, token in enumerate(tokens):
            input_tokens.append(token)
            segment_ids.append(0)
            label_ids.append(labels[i])
            feature_blocks.append(features[i])
            chars_blocks.append(chars[i])

        for token in tokens_marked:
            input_tokens_marked.append(token)

        input_tokens.append(self.tokenizer.sep_token)
        input_tokens_marked.append(self.tokenizer.sep_token)
        segment_ids.append(0)
        label_ids.append("<PAD>")
        chars_blocks.append(self.empty_char_vector)
        feature_blocks.append(self.empty_features_vector)

        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(self.tokenizer.pad_token_id)
            input_mask.append(self.tokenizer.pad_token_id)
            segment_ids.append(0)
            label_ids.append("<PAD>")
            chars_blocks.append(self.empty_char_vector)
            feature_blocks.append(self.empty_features_vector)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(chars_blocks) == max_seq_length
        assert len(feature_blocks) == max_seq_length

        return input_ids, input_mask, segment_ids, chars_blocks, feature_blocks, label_ids, input_tokens_marked
