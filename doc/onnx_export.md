# ONNX Export

DeLFT supports exporting trained models to ONNX format for inference in Java or other runtimes (Python, C++, etc.).

## Overview

The ONNX export creates portable model files that can be loaded without Python/PyTorch dependencies. This enables:

- **Java integration**: Run models in GROBID or other Java applications
- **Cross-platform deployment**: Use ONNX Runtime on various platforms
- **Reduced dependencies**: No need for PyTorch at inference time

## Supported Model Types

### Sequence Labelling

| Architecture | Status | Notes |
|--------------|--------|-------|
| `BidLSTM_CRF` | ✅ Supported | Exports encoder + CRF params |
| `BidLSTM_CRF_FEATURES` | ✅ Supported | Includes feature embeddings |
| `BidLSTM_ChainCRF` | ✅ Supported | Alternative CRF implementation |
| `BidLSTM_ChainCRF_FEATURES` | ✅ Supported | ChainCRF with features |
| `BERT` | ✅ Supported | Transformer + softmax, ~400MB |
| `BERT_CRF` | ✅ Supported | Transformer + CRF, ~400MB |
| `BERT_ChainCRF` | ✅ Supported | Transformer + ChainCRF, ~400MB |
| Other BERT variants | ⚠️ Experimental | May work, not fully tested |

### Text Classification

| Architecture | Status | Notes |
|--------------|--------|-------|
| `gru` | ✅ Supported | Two-layer BiGRU |
| `gru_simple` | ✅ Supported | Single-layer BiGRU |
| `lstm` | ✅ Supported | LSTM classifier |
| `bidLstm_simple` | ✅ Supported | BiLSTM classifier |
| `cnn`, `cnn2`, `cnn3` | ✅ Supported | CNN variants |
| `gru_lstm` | ✅ Supported | Mixed GRU+LSTM |
| `lstm_cnn` | ✅ Supported | LSTM+CNN hybrid |
| `dpcnn` | ✅ Supported | Deep Pyramid CNN |
| `bert` | ❌ Not supported | Different input format |

## Usage

### Sequence Labelling Export

```bash
python -m delft.applications.onnx_export header \
    --architecture BidLSTM_CRF_FEATURES \
    --output exported_models
```

**Output files:**
- `encoder.onnx` - BiLSTM encoder (outputs emission scores)
- `crf_params.json` - CRF transition matrices for Viterbi decode
- `vocab.json` - Character and label vocabularies
- `config.json` - Model configuration

### Transformer Export (BERT)

```bash
python -m delft.applications.onnx_export header \
    --architecture BERT_CRF \
    --output exported_models
```

**Output files:**
- `encoder.onnx` - Transformer encoder (~400MB, embeddings included)
- `crf_params.json` - CRF transition matrices (for CRF variants)
- `tokenizer/` - HuggingFace tokenizer files for Java
- `config.json` - Model configuration with label mappings

### Text Classification Export

```bash
python -m delft.applications.onnx_export dataseer-binary \
    --model-type classification \
    --architecture gru \
    --output exported_models
```

**Output files:**
- `classifier.onnx` - Classification model (outputs logits)
- `labels.json` - Class label mappings
- `config.json` - Model configuration (includes `activationFunction`)

## Limitations

### General Limitations

1. **Word embeddings not included** (BiLSTM models): BiLSTM models expect pre-computed word embeddings as input. Load embeddings separately (e.g., from LMDB). Transformer models include embeddings in the ONNX file.

2. **Dynamic shapes**: Models are exported with static batch size for compatibility. Sequence length is dynamic.

3. **opset version 14**: Models use ONNX opset 14 for broad compatibility.

### Sequence Labelling Limitations

1. **CRF decode is external**: The ONNX model outputs raw emission scores. CRF Viterbi decoding must be implemented separately using `crf_params.json`.

2. **Feature models require feature input**: Models with `_FEATURES` suffix expect a features tensor in addition to word/char embeddings.

### Text Classification Limitations

1. **BERT not supported**: Transformer-based classifiers use different input format (tokenized input_ids) and are not currently supported for ONNX export.

2. **Activation applied externally**: The model outputs raw logits. Apply `sigmoid` (config specifies `"activationFunction": "sigmoid"`) to get probabilities.

3. **Multi-label by default**: Current models use BCEWithLogitsLoss and sigmoid activation (multi-label style). For single-label classification with softmax, model changes would be needed.

## Java Integration

A Java inference library is available in `java/delft-onnx/`.

### BiLSTM Models

Requires word embeddings from LMDB.

### Transformer Models

For BERT-based models, Java integration requires:

1. **Tokenizer**: Use [DJL HuggingFace tokenizers](https://github.com/deepjavalibrary/djl) or [tokenizers-java](https://github.com/huggingface/tokenizers/tree/main/bindings/java) to load the exported `tokenizer/` files
2. **Sub-token alignment**: Map GrobidAnalyzer tokens to BERT sub-tokens
3. **ONNX Runtime**: Load `encoder.onnx` with ONNX Runtime Java
4. **CRF Decode**: Apply Viterbi using `crf_params.json` (for CRF variants)
5. **Label alignment**: Map sub-token predictions back to original tokens

## Troubleshooting

### Model verification fails
Ensure the `onnx` package is installed: `pip install onnx`

### Output shape mismatch
Check that input tensor shapes match expected dimensions in `config.json`.

### Missing features input
For `_FEATURES` models, ensure you provide the features tensor with correct shape.
