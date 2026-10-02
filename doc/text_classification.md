## Text classification

### Available models

All the following models includes Dropout, Pooling and Dense layers with hyperparameters tuned for reasonable performance across standard text classification tasks. If necessary, they are good basis for further performance tuning.

* `bert`: a transformer classifier to fine-tune, to be instantiated by any BERT pre-trained model or transformers available on HuggingFace Hub (we have tested various BERT and RoBERTa flavors). Given no transformer (`transformer_name`, `--transformer`), it uses `bert-base-uncased`.
* `gru`: two layers Bidirectional GRU
* `gru_simple`: one layer Bidirectional GRU
* `gru_lstm`: one layer Bidirectional GRU followed by a Bidirectional LSTM
* `lstm`: an LSTM layer followed by an Attention layer
* `bidLstm_simple`: a Bidirectional LSTM layer followed by an Attention layer
* `cnn`: convolutional layers followed by a GRU
* `cnn2`: convolutional variant with a GRU
* `cnn3`: GRU followed by convolutional layers with pooling
* `lstm_cnn`: LSTM followed by convolutional layers
* `dpcnn`: Deep Pyramid Convolutional Neural Networks (experimental)

The authoritative list is `MODEL_REGISTRY` in `delft/textClassification/models.py`.

Note: a training set too big for memory can be read from disk, one text at a time. `load_texts_and_classes(path, in_memory=False)` in `delft.textClassification.reader` gives, in place of the array of texts, a `TextsOnDisk` that keeps the offsets of the lines of the tab-separated file alone and reads a text when the data loader asks for it. It is used like the array of texts: shuffling and splitting it into training and validation sets keep the texts on disk.

Note: by default the first 300 tokens of the text to be classified are used, which is largely enough for any _short text_ classification tasks and works fine with low profile GPU (for instance GeForce GTX 1050 Ti with 4 GB memory). For taking into account a larger portion of the text, modify the config model parameter `maxlen`. However, using more than 1000 tokens for instance requires a modern GPU with enough memory (e.g. 10 GB).

### Contextual embeddings read the text as a whole

With [contextual embeddings from a frozen transformer](embeddings.md#contextual-embeddings-from-a-frozen-transformer) (`--embedding scibert-contextual`, alone or in a stack), the transformer is given the tokens of a text one by one, a space before each. With `--whole-text-tokenization` (`whole_text_tokenization=True` for `Classifier`), the tokens are joined back into a text with the usual spacing of punctuation and the transformer reads that text, which is what a SentencePiece or byte-level BPE tokenizer was pretrained on: see [whole text sub-tokenization](sequence_labeling.md#whole-text-sub-tokenization). The option is saved with the model, and the vectors have a cache of their own. It has no effect on the `bert` architecture, which reads the text itself, nor on static embeddings.

### Training again with the same result

`delft.utilities.Utilities.set_random_seed(seed)` seeds Python, NumPy and PyTorch, so that the split of the data, the initial weights and the order of the batches are the same from a run to the next. The applications call it with their `--seed` option; without it every run draws its own.

### Training over folds

A classifier can be trained over several folds (`fold_number` of `Classifier`, `--fold-count` of the applications), which gives an ensemble: usually more accurate than a single model, and as many times slower to train and to run.

```python
from delft.textClassification.wrapper import Classifier

classifier = Classifier(
    "my-model", architecture="gru", embeddings_name="glove-840B", list_classes=["yes", "no"], fold_number=10
)
classifier.train(texts, classes)  # or classifier.train_nfold(texts, classes)
classifier.eval(test_texts, test_classes)
classifier.save()
```

How it works:

- **Training.** The training texts are shuffled, then cut into `fold_number` folds of the same size, the last one taking the texts the division leaves over. One model is trained per fold, on the texts of all the other folds. With early stopping, the fold itself is the validation set of its model: every model is validated on texts it was not trained on, and every text serves once as validation. Without early stopping the fold is simply left out.
- **Classifying.** `predict` and `eval` run every fold model, and the probability of a class is the geometric mean of the probabilities the models give it.
- **Saving and loading.** `save` writes the weights of every fold model next to the one configuration, as `model_fold0.safetensors`, `model_fold1.safetensors` and so on; `load` reads the number of folds from the configuration and loads them all, in whichever format they were saved (`model_weights_fold0.pt` for pickled weights). The weights a previous training left in the directory, for a single model or for another number of folds, are removed.
- **Incremental training.** After a `load`, `train(..., incremental=True)` goes on with the training of every fold model on new folds of the given texts.
- **Transformers.** With the `bert` architecture every fold is a whole transformer. On a GPU the fold models are kept in main memory and moved to the GPU one at a time, for the time each runs, since several may not fit there together; expect a classification to take as many times longer as there are folds.

Sequence labelling uses folds differently: there the fold models are compared on the evaluation set and the best one is the model kept.
