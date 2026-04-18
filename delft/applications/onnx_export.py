"""
ONNX Export for DeLFT sequence labeling models.

Exports trained BiLSTM+CRF models to ONNX format for inference in Java/other runtimes.

Supported architectures:
- BidLSTM_CRF
- BidLSTM_CRF_FEATURES
- BidLSTM_ChainCRF
- BidLSTM_ChainCRF_FEATURES

Usage:
    python -m delft.applications.onnx_export --model MODEL_NAME --output OUTPUT_DIR
"""

import os
import json
import argparse

import torch
import numpy as np

from delft.sequenceLabelling.wrapper import Sequence


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
    if hasattr(crf, "crf"):
        # Using pytorch-crf wrapper (standard CRF class)
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
    max_seq_length: int = 512,
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
        max_seq_length: Maximum sequence length for ONNX model
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


def main():
    parser = argparse.ArgumentParser(description="Export DeLFT model to ONNX format")
    parser.add_argument("--model", required=True, help="Name of the model to export")
    parser.add_argument(
        "--output", required=True, help="Output directory for exported files"
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)",
    )
    parser.add_argument(
        "--max-char-length",
        type=int,
        default=30,
        help="Maximum character length per token (default: 30)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Custom model path (default: data/models/sequenceLabelling/)",
    )

    args = parser.parse_args()

    export_to_onnx(
        model_name=args.model,
        output_dir=args.output,
        max_seq_length=args.max_seq_length,
        max_char_length=args.max_char_length,
        model_path=args.model_path,
    )


if __name__ == "__main__":
    main()
