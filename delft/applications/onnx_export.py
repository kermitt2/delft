"""
ONNX Export for DeLFT models.

Exports trained models to ONNX format for inference in Java/other runtimes.

Supported model types:
1. Sequence Labelling (BiLSTM+CRF):
   - BidLSTM_CRF, BidLSTM_CRF_FEATURES
   - BidLSTM_ChainCRF, BidLSTM_ChainCRF_FEATURES

2. Text Classification (embedding-based):
   - gru, gru_simple, gru_lstm
   - lstm, bidLstm_simple, lstm_cnn
   - cnn, cnn2, cnn3, dpcnn

Usage:
    # Sequence labelling (default)
    python -m delft.applications.onnx_export header --architecture BidLSTM_CRF

    # Text classification
    python -m delft.applications.onnx_export dataseer-binary --model-type classification --architecture gru
"""

import os
import json
import argparse

import torch
import numpy as np

from delft.sequenceLabelling.wrapper import Sequence
from delft.textClassification import Classifier


class EncoderWrapper(torch.nn.Module):
    """
    Wrapper that extracts encoder (without CRF decode) for ONNX export.

    Returns emission scores that can be decoded by external CRF implementation.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, word_input, char_input, features_input=None):
        """Run encoder to get emissions."""
        inputs = {
            "word_input": word_input,
            "char_input": char_input,
        }
        if features_input is not None:
            inputs["features_input"] = features_input

        # Get outputs (without labels, so no CRF loss computation)
        outputs = self.model(inputs, labels=None)
        return outputs["logits"]


class ClassifierWrapper(torch.nn.Module):
    """
    Wrapper for text classification models for ONNX export.

    Takes pre-computed word embeddings and returns class logits.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, embeddings):
        """Run classifier to get logits.

        Args:
            embeddings: Word embeddings tensor [batch, seq_len, embed_size]

        Returns:
            logits: Classification logits [batch, num_classes]
        """
        outputs = self.model(embeddings, labels=None)
        return outputs["logits"]


class TransformerEncoderWrapper(torch.nn.Module):
    """
    Wrapper for transformer-based sequence labelling models for ONNX export.

    Takes tokenized input_ids/attention_mask and returns emission scores.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.accepts_token_type_ids = getattr(model, 'accepts_token_type_ids', True)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Run transformer encoder to get emissions.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Optional token type IDs [batch, seq_len]

        Returns:
            emissions: Logits/emissions [batch, seq_len, num_tags]
        """
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if self.accepts_token_type_ids and token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        outputs = self.model(inputs, labels=None)
        return outputs["logits"]


def export_crf_params(model, output_path: str):
    """
    Export CRF transition matrices to JSON.

    Args:
        model: The model containing CRF layer
        output_path: Path to save JSON file
    """
    crf = model.crf

    params = {}

    # Handle different CRF implementations
    # Standard Contract: crf_params.json MUST contain "transitions" in [from_tag][to_tag] orientation.
    # Java Code (CRFDecoder.java) iterates as transitions[prevTag][currentTag].
    
    if hasattr(crf, "crf"):
        # Using pytorch-crf wrapper (standard CRF class)
        # pytorch-crf stores transitions as [from_tag, to_tag]
        # This already matches Java's expected [from_tag][to_tag] format.
        inner_crf = crf.crf
        params["transitions"] = inner_crf.transitions.detach().cpu().numpy().tolist()
        params["startTransitions"] = (
            inner_crf.start_transitions.detach().cpu().numpy().tolist()
        )
        params["endTransitions"] = (
            inner_crf.end_transitions.detach().cpu().numpy().tolist()
        )
    elif hasattr(crf, "U"):
        # ChainCRF - uses U (transitions), b_start, b_end naming
        params["transitions"] = crf.U.detach().cpu().numpy().tolist()
        params["startTransitions"] = crf.b_start.detach().cpu().numpy().tolist()
        params["endTransitions"] = crf.b_end.detach().cpu().numpy().tolist()
    elif hasattr(crf, "transitions"):
        # Custom CRF fallback
        params["transitions"] = crf.transitions.detach().cpu().numpy().tolist()
        params["startTransitions"] = (
            crf.start_transitions.detach().cpu().numpy().tolist()
        )
        params["endTransitions"] = crf.end_transitions.detach().cpu().numpy().tolist()
    else:
        raise ValueError("Unknown CRF implementation")

    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)

    print(f"Exported CRF params to {output_path}")
    print(
        f"  Transitions shape: {len(params['transitions'])}x{len(params['transitions'][0])}"
    )


def export_vocab(preprocessor, model_config, output_path: str):
    """
    Export vocabularies and label mappings.

    Args:
        preprocessor: The preprocessor with vocab_char, vocab_tag
        model_config: Model configuration
        output_path: Path to save JSON file
    """
    vocab = {
        "charVocab": preprocessor.vocab_char,
        "tagVocab": preprocessor.vocab_tag,
        "tagIndex": {str(k): v for k, v in preprocessor.indice_tag.items()},
        "maxCharLength": preprocessor.max_char_length,
        "returnChars": preprocessor.return_chars,
    }

    # Add feature info if available
    if (
        hasattr(preprocessor, "feature_preprocessor")
        and preprocessor.feature_preprocessor
    ):
        fp = preprocessor.feature_preprocessor
        vocab["featuresIndices"] = (
            list(fp.features_indices) if fp.features_indices else None
        )
        vocab["featuresVocabularySize"] = fp.features_vocabulary_size
        vocab["featuresMapToIndex"] = fp.features_map_to_index

    with open(output_path, "w") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)

    print(f"Exported vocab to {output_path}")
    print(f"  Char vocab size: {len(vocab['charVocab'])}")
    print(f"  Tag vocab size: {len(vocab['tagVocab'])}")


def export_config(model_config, training_config, output_path: str):
    """
    Export model configuration for Java runtime.
    """
    config = {
        "modelName": model_config.model_name,
        "architecture": model_config.architecture,
        "wordEmbeddingSize": model_config.word_embedding_size,
        "charEmbeddingSize": model_config.char_embedding_size,
        "numCharLstmUnits": model_config.num_char_lstm_units,
        "numWordLstmUnits": model_config.num_word_lstm_units,
        "maxSequenceLength": model_config.max_sequence_length,
        "embeddingsName": model_config.embeddings_name,
    }

    # Add features config if present
    if hasattr(model_config, "features_indices") and model_config.features_indices:
        config["featuresIndices"] = list(model_config.features_indices)
        config["featuresEmbeddingSize"] = model_config.features_embedding_size
        config["featuresLstmUnits"] = model_config.features_lstm_units
        config["featuresVocabularySize"] = model_config.features_vocabulary_size

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Exported config to {output_path}")


def export_classification_config(model_config, output_path: str):
    """
    Export text classification model configuration for Java runtime.
    """
    config = {
        "modelName": model_config.model_name,
        "architecture": model_config.architecture,
        "wordEmbeddingSize": model_config.word_embedding_size,
        "maxlen": model_config.maxlen,
        "numClasses": len(model_config.list_classes),
        "embeddingsName": model_config.embeddings_name,
        # All current models use BCEWithLogitsLoss (sigmoid for multi-label)
        # If models with CrossEntropyLoss are added, this should be "softmax"
        "activationFunction": "sigmoid",
    }

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Exported classification config to {output_path}")



def export_class_labels(model_config, output_path: str):
    """
    Export class labels for text classification model.
    """
    labels = {
        "labels": model_config.list_classes,
        "labelToIndex": {label: idx for idx, label in enumerate(model_config.list_classes)},
        "indexToLabel": {idx: label for idx, label in enumerate(model_config.list_classes)},
    }

    with open(output_path, "w") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    print(f"Exported class labels to {output_path}")
    print(f"  Number of classes: {len(model_config.list_classes)}")


def export_tokenizer(tokenizer, output_dir: str):
    """
    Export HuggingFace tokenizer files for Java runtime.

    Exports:
        - vocab.txt or vocab.json (depending on tokenizer type)
        - tokenizer_config.json
        - special_tokens_map.json

    Args:
        tokenizer: HuggingFace tokenizer instance
        output_dir: Directory to save tokenizer files
    """
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    print(f"Exported tokenizer to {output_dir}")
    
    # List exported files
    for f in os.listdir(output_dir):
        print(f"  - {f}")


def export_transformer_config(model_config, preprocessor, accepts_token_type_ids: bool, output_path: str):
    """
    Export transformer model configuration for Java runtime.

    Args:
        model_config: Model configuration
        preprocessor: The preprocessor with vocab_tag
        accepts_token_type_ids: Whether the model accepts token_type_ids
        output_path: Path to save JSON file
    """
    config = {
        "modelName": model_config.model_name,
        "architecture": model_config.architecture,
        "transformerName": model_config.transformer_name,
        "maxSequenceLength": model_config.max_sequence_length,
        "useCRF": "CRF" in model_config.architecture or "ChainCRF" in model_config.architecture,
        "useChainCRF": "ChainCRF" in model_config.architecture,
        "useFeatures": "FEATURES" in model_config.architecture,
        "useChar": "CHAR" in model_config.architecture,
        "acceptsTokenTypeIds": accepts_token_type_ids,
    }

    # Add label mappings
    if hasattr(preprocessor, 'vocab_tag'):
        config["labelVocab"] = preprocessor.vocab_tag
        config["labelIndex"] = {str(k): v for k, v in preprocessor.indice_tag.items()}
        config["numLabels"] = len(preprocessor.vocab_tag)

    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Exported transformer config to {output_path}")


def export_embeddings(embeddings, output_path: str, vocab_words: list = None):
    """
    Export word embeddings in a simple binary format for Java.

    Format:
        - 4 bytes: embedding dimension (int, little-endian)
        - 4 bytes: number of words (int, little-endian)
        - For each word:
            - 4 bytes: word length (int, little-endian)
            - N bytes: word (UTF-8)
            - D * 4 bytes: embedding vector (D floats, little-endian)

    Args:
        embeddings: The Embeddings object from DeLFT
        output_path: Path to save binary file
        vocab_words: Optional list of words to export (if None, exports common words)
    """
    import struct

    # Get embedding dimension
    embed_dim = embeddings.embed_size

    # If no vocab specified, get some common words
    if vocab_words is None:
        # We'll export only words that were seen during use
        # For now, just export a small test set
        vocab_words = []

    # Filter to only words that exist in embeddings
    valid_words = []
    embeddings_data = []

    for word in vocab_words:
        emb = embeddings.get_word_vector(word)
        if emb is not None and not np.all(emb == 0):
            valid_words.append(word)
            embeddings_data.append(emb)

    print(f"Exporting {len(valid_words)} embeddings to {output_path}")

    with open(output_path, "wb") as f:
        # Write header
        f.write(struct.pack("<I", embed_dim))  # embedding dimension
        f.write(struct.pack("<I", len(valid_words)))  # word count

        # Write each word + embedding
        for word, emb in zip(valid_words, embeddings_data):
            word_bytes = word.encode("utf-8")
            f.write(struct.pack("<I", len(word_bytes)))  # word length
            f.write(word_bytes)  # word
            f.write(emb.astype(np.float32).tobytes())  # embedding

    print(f"  Embedding dimension: {embed_dim}")
    print(f"  Words exported: {len(valid_words)}")


def export_to_onnx(
    model_name: str,
    output_dir: str,
    max_seq_length: int = None,
    max_char_length: int = 30,
    model_path: str = None,
):
    """
    Export a trained DeLFT model to ONNX format.

    Creates:
        - encoder.onnx: The BiLSTM encoder (emissions output)
        - crf_params.json: CRF transition matrices
        - vocab.json: Character and label vocabularies
        - config.json: Model configuration

    Args:
        model_name: Name of the model to export
        output_dir: Directory to save exported files
        max_seq_length: Maximum sequence length for ONNX model (default: from model config)
        max_char_length: Maximum character length per token
        model_path: Optional custom model path
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model: {model_name}")

    # Load the model
    model_wrapper = Sequence(model_name)
    if model_path:
        model_wrapper.load(dir_path=model_path)
    else:
        model_wrapper.load()

    model = model_wrapper.model
    preprocessor = model_wrapper.p
    model_config = model_wrapper.model_config
    training_config = model_wrapper.training_config

    # Use model's configured max_sequence_length if not explicitly provided
    if max_seq_length is None:
        max_seq_length = model_config.max_sequence_length
        print(f"Using max_sequence_length from model config: {max_seq_length}")

    # Verify architecture - list supported architectures
    arch = model_config.architecture
    supported_archs = [
        "BidLSTM_CRF",
        "BidLSTM_CRF_FEATURES",
        "BidLSTM_ChainCRF",
        "BidLSTM_ChainCRF_FEATURES",
    ]
    if arch not in supported_archs:
        print(
            f"Warning: Architecture {arch} may not be fully supported. Proceeding anyway."
        )

    print(f"Architecture: {arch}")

    # Set model to eval mode
    model.eval()

    # Create encoder wrapper
    encoder = EncoderWrapper(model)
    encoder.eval()

    # Determine input shapes
    word_emb_size = model_config.word_embedding_size
    max_char = max_char_length
    batch_size = 1
    seq_len = max_seq_length

    # Create dummy inputs
    dummy_word_input = torch.randn(batch_size, seq_len, word_emb_size)
    dummy_char_input = torch.zeros(batch_size, seq_len, max_char, dtype=torch.long)

    # Input/output names
    input_names = ["word_input", "char_input"]
    output_names = ["emissions"]

    # Dynamic axes for variable batch and sequence length
    dynamic_axes = {
        "word_input": {0: "batch_size", 1: "seq_length"},
        "char_input": {0: "batch_size", 1: "seq_length"},
        "emissions": {0: "batch_size", 1: "seq_length"},
    }

    # Handle features input for FEATURES architectures
    if arch in ["BidLSTM_CRF_FEATURES", "BidLSTM_ChainCRF_FEATURES"]:
        num_features = (
            len(model_config.features_indices) if model_config.features_indices else 1
        )
        dummy_features_input = torch.zeros(
            batch_size, seq_len, num_features, dtype=torch.long
        )
        input_names.append("features_input")
        dynamic_axes["features_input"] = {0: "batch_size", 1: "seq_length"}
        dummy_inputs = (dummy_word_input, dummy_char_input, dummy_features_input)
    else:
        dummy_inputs = (dummy_word_input, dummy_char_input)

    # Export to ONNX (using static shapes for compatibility)
    onnx_path = os.path.join(output_dir, "encoder.onnx")
    print(f"Exporting to ONNX: {onnx_path}")
    print(
        f"  Static shapes: batch=1, seq_len={seq_len}, word_emb={word_emb_size}, max_char={max_char}"
    )

    # Use dynamo=False for older-style export that's more compatible
    torch.onnx.export(
        encoder,
        dummy_inputs,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        verbose=False,
        dynamo=False,  # Use legacy export for better compatibility
    )

    print("ONNX model exported successfully")

    # Export CRF params
    crf_path = os.path.join(output_dir, "crf_params.json")
    export_crf_params(model, crf_path)

    # Export vocab
    vocab_path = os.path.join(output_dir, "vocab.json")
    export_vocab(preprocessor, model_config, vocab_path)

    # Export config
    config_path = os.path.join(output_dir, "config.json")
    export_config(model_config, training_config, config_path)

    # Verify ONNX model
    try:
        import onnx

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed")
    except ImportError:
        print("Warning: onnx package not installed, skipping verification")
    except Exception as e:
        print(f"Warning: ONNX verification failed: {e}")

    print(f"\nExport complete! Files in {output_dir}:")
    for f in os.listdir(output_dir):
        print(f"  - {f}")

    return output_dir


def export_classification_to_onnx(
    model_name: str,
    output_dir: str,
    max_seq_length: int = None,
    model_path: str = None,
):
    """
    Export a trained text classification model to ONNX format.

    Creates:
        - classifier.onnx: The classification model (embeddings → logits)
        - config.json: Model configuration
        - labels.json: Class labels mapping

    Args:
        model_name: Name of the model to export
        output_dir: Directory to save exported files
        max_seq_length: Maximum sequence length for ONNX model (default: from model config)
        model_path: Optional custom model path
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading classification model: {model_name}")

    # Load the model
    model_wrapper = Classifier(model_name)
    if model_path:
        model_wrapper.load(dir_path=model_path)
    else:
        model_wrapper.load()

    model = model_wrapper.model
    model_config = model_wrapper.model_config

    # Check architecture is supported
    arch = model_config.architecture
    if arch == "bert":
        raise ValueError(
            f"BERT architecture is not supported for ONNX export. "
            f"Use one of: {CLASSIFICATION_ARCHITECTURES}"
        )

    # Use model's configured maxlen if not explicitly provided
    if max_seq_length is None:
        max_seq_length = model_config.maxlen
        print(f"Using maxlen from model config: {max_seq_length}")

    print(f"Architecture: {arch}")

    # Set model to eval mode
    model.eval()

    # Create classifier wrapper
    classifier = ClassifierWrapper(model)
    classifier.eval()

    # Determine input shapes
    word_emb_size = model_config.word_embedding_size
    batch_size = 1
    seq_len = max_seq_length

    # Create dummy input (pre-computed embeddings)
    dummy_embeddings = torch.randn(batch_size, seq_len, word_emb_size)

    # Input/output names
    input_names = ["embeddings"]
    output_names = ["logits"]

    # Dynamic axes for variable batch and sequence length
    dynamic_axes = {
        "embeddings": {0: "batch_size", 1: "seq_length"},
        "logits": {0: "batch_size"},
    }

    # Export to ONNX
    onnx_path = os.path.join(output_dir, "classifier.onnx")
    print(f"Exporting to ONNX: {onnx_path}")
    print(f"  Static shapes: batch=1, seq_len={seq_len}, word_emb={word_emb_size}")

    torch.onnx.export(
        classifier,
        dummy_embeddings,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        verbose=False,
        dynamo=False,  # Use legacy export for better compatibility
    )

    print("ONNX model exported successfully")

    # Export config
    config_path = os.path.join(output_dir, "config.json")
    export_classification_config(model_config, config_path)

    # Export class labels
    labels_path = os.path.join(output_dir, "labels.json")
    export_class_labels(model_config, labels_path)

    # Verify ONNX model
    try:
        import onnx

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed")
    except ImportError:
        print("Warning: onnx package not installed, skipping verification")
    except Exception as e:
        print(f"Warning: ONNX verification failed: {e}")

    print(f"\nExport complete! Files in {output_dir}:")
    for f in os.listdir(output_dir):
        print(f"  - {f}")

    return output_dir


def export_transformer_to_onnx(
    model_name: str,
    output_dir: str,
    max_seq_length: int = 512,
    model_path: str = None,
):
    """
    Export a trained transformer-based sequence labelling model to ONNX format.

    Creates:
        - encoder.onnx: Transformer encoder (input_ids → emissions), ~400MB
        - crf_params.json: CRF parameters (if applicable)
        - tokenizer/: HuggingFace tokenizer files for Java
        - config.json: Model configuration with label mappings

    Args:
        model_name: Name of the model to export
        output_dir: Directory to save exported files
        max_seq_length: Maximum sequence length (default: 512)
        model_path: Optional custom model path
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading transformer model: {model_name}")

    # Load the model
    model_wrapper = Sequence(model_name)
    if model_path:
        model_wrapper.load(dir_path=model_path)
    else:
        model_wrapper.load()

    model = model_wrapper.model
    preprocessor = model_wrapper.p
    model_config = model_wrapper.model_config

    # Verify it's a transformer architecture
    arch = model_config.architecture
    if arch not in ARCHITECTURES_TRANSFORMERS:
        raise ValueError(
            f"Not a transformer architecture: {arch}. "
            f"Must be one of: {ARCHITECTURES_TRANSFORMERS}"
        )

    print(f"Architecture: {arch}")
    print(f"Transformer: {model_config.transformer_name}")

    # Set model to eval mode
    model.eval()

    # Create transformer encoder wrapper
    encoder = TransformerEncoderWrapper(model)
    encoder.eval()

    # Get whether model accepts token_type_ids
    accepts_token_type_ids = encoder.accepts_token_type_ids
    print(f"Accepts token_type_ids: {accepts_token_type_ids}")

    # Create dummy inputs
    batch_size = 1
    seq_len = max_seq_length
    dummy_input_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    
    # Prepare inputs and names based on model requirements
    if accepts_token_type_ids:
        dummy_token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        dummy_inputs = (dummy_input_ids, dummy_attention_mask, dummy_token_type_ids)
        input_names = ["input_ids", "attention_mask", "token_type_ids"]
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_length"},
            "attention_mask": {0: "batch_size", 1: "seq_length"},
            "token_type_ids": {0: "batch_size", 1: "seq_length"},
            "emissions": {0: "batch_size", 1: "seq_length"},
        }
    else:
        dummy_inputs = (dummy_input_ids, dummy_attention_mask)
        input_names = ["input_ids", "attention_mask"]
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "seq_length"},
            "attention_mask": {0: "batch_size", 1: "seq_length"},
            "emissions": {0: "batch_size", 1: "seq_length"},
        }

    output_names = ["emissions"]

    # Export to ONNX
    onnx_path = os.path.join(output_dir, "encoder.onnx")
    print(f"Exporting transformer to ONNX: {onnx_path}")
    print(f"  Static shapes for export: batch=1, seq_len={seq_len}")
    print(f"  Note: Model includes transformer weights (~400MB)")

    torch.onnx.export(
        encoder,
        dummy_inputs,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
        verbose=False,
        dynamo=False,  # Use legacy export for better compatibility
    )

    print("ONNX transformer model exported successfully")

    # Export CRF params if applicable
    if hasattr(model, 'crf'):
        crf_path = os.path.join(output_dir, "crf_params.json")
        export_crf_params(model, crf_path)
    else:
        print("No CRF layer found (softmax output model)")

    # Export tokenizer
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    if hasattr(preprocessor, 'tokenizer') and preprocessor.tokenizer is not None:
        export_tokenizer(preprocessor.tokenizer, tokenizer_dir)
    else:
        print("Warning: No tokenizer found in preprocessor, skipping tokenizer export")
        print("  You may need to load the tokenizer separately using the transformer name")

    # Export config
    config_path = os.path.join(output_dir, "config.json")
    export_transformer_config(model_config, preprocessor, accepts_token_type_ids, config_path)

    # Verify ONNX model
    try:
        import onnx

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed")
    except ImportError:
        print("Warning: onnx package not installed, skipping verification")
    except Exception as e:
        print(f"Warning: ONNX verification failed: {e}")

    print(f"\nExport complete! Files in {output_dir}:")
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            print(f"  - {item}/")
            for sub_item in os.listdir(item_path):
                print(f"      - {sub_item}")
        else:
            # Show file size for ONNX file
            if item.endswith('.onnx'):
                size_mb = os.path.getsize(item_path) / (1024 * 1024)
                print(f"  - {item} ({size_mb:.1f} MB)")
            else:
                print(f"  - {item}")

    return output_dir


# Model lists matching grobidTagger.py
MODEL_LIST = [
    "affiliation-address",
    "citation",
    "date",
    "header",
    "name-citation",
    "name-header",
    "software",
    "figure",
    "table",
    "reference-segmenter",
    "segmentation",
    "funding-acknowledgement",
    "patent-citation",
]

# Classification model list
CLASSIFICATION_MODEL_LIST = [
    "dataseer-binary",
    "dataseer-first",
    "dataseer-reuse",
    "citation",
    "license",
    "software",
    "software-context",
    "toxic-comment",
]

# Classification architectures (embedding-based only, BERT not supported)
CLASSIFICATION_ARCHITECTURES = [
    "gru",
    "gru_simple",
    "gru_lstm",
    "lstm",
    "bidLstm_simple",
    "lstm_cnn",
    "cnn",
    "cnn2",
    "cnn3",
    "dpcnn",
]

ARCHITECTURES_WORD_EMBEDDINGS = [
    "BidLSTM",
    "BidLSTM_CRF",
    "BidLSTM_ChainCRF",
    "BidLSTM_CNN_CRF",
    "BidGRU_CRF",
    "BidLSTM_CNN",
    "BidLSTM_CRF_CASING",
    "BidLSTM_CRF_FEATURES",
    "BidLSTM_ChainCRF_FEATURES",
]

ARCHITECTURES_TRANSFORMERS = [
    "BERT",
    "BERT_FEATURES",
    "BERT_CRF",
    "BERT_ChainCRF",
    "BERT_CRF_FEATURES",
    "BERT_ChainCRF_FEATURES",
    "BERT_CRF_CHAR",
    "BERT_CRF_CHAR_FEATURES",
]

ARCHITECTURES = ARCHITECTURES_WORD_EMBEDDINGS + ARCHITECTURES_TRANSFORMERS


def main():
    parser = argparse.ArgumentParser(description="Export DeLFT model to ONNX format")
    parser.add_argument("model", help="Name of the model (e.g., header, date, dataseer-binary)")
    parser.add_argument(
        "--architecture",
        required=True,
        help="Model architecture",
    )
    parser.add_argument(
        "--model-type",
        default="sequence",
        choices=["sequence", "classification"],
        help="Type of model: 'sequence' for sequence labelling (default), "
        "'classification' for text classification",
    )
    parser.add_argument(
        "--output",
        default="exported_models",
        help="Base output directory (default: exported_models). "
        "Model will be saved in {output}/{model}-{architecture}.onnx/",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Maximum sequence length (default: from model config)",
    )
    parser.add_argument(
        "--max-char-length",
        type=int,
        default=30,
        help="Maximum character length per token (default: 30, sequence labelling only)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Custom model path",
    )

    args = parser.parse_args()

    if args.model_type == "classification":
        # Text classification export
        if args.architecture not in CLASSIFICATION_ARCHITECTURES:
            raise ValueError(
                f"Unknown classification architecture: {args.architecture}. "
                f"Must be one of: {CLASSIFICATION_ARCHITECTURES}"
            )

        # Construct model name (classification models use model_architecture format)
        model_name = args.model + "_" + args.architecture

        # Construct output directory
        output_subdir = f"{args.model}-{args.architecture}.onnx"
        output_dir = os.path.join(args.output, output_subdir)

        export_classification_to_onnx(
            model_name=model_name,
            output_dir=output_dir,
            max_seq_length=args.max_seq_length,
            model_path=args.model_path,
        )
    else:
        # Sequence labelling export (default)
        if args.architecture not in ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture: {args.architecture}. Must be one of: {ARCHITECTURES}"
            )

        # Construct model name similar to grobidTagger.py
        model_name = "grobid-" + args.model + "-" + args.architecture

        # Construct output directory
        output_subdir = f"{args.model}-{args.architecture}.onnx"
        output_dir = os.path.join(args.output, output_subdir)

        # Route to appropriate export function based on architecture type
        if args.architecture in ARCHITECTURES_TRANSFORMERS:
            # Transformer-based models (BERT, etc.)
            export_transformer_to_onnx(
                model_name=model_name,
                output_dir=output_dir,
                max_seq_length=args.max_seq_length or 512,
                model_path=args.model_path,
            )
        else:
            # Word embedding-based models (BiLSTM, etc.)
            export_to_onnx(
                model_name=model_name,
                output_dir=output_dir,
                max_seq_length=args.max_seq_length,
                max_char_length=args.max_char_length,
                model_path=args.model_path,
            )


if __name__ == "__main__":
    main()
