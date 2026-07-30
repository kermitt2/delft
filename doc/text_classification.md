## Text classification

### Available models

All the following models includes Dropout, Pooling and Dense layers with hyperparameters tuned for reasonable performance across standard text classification tasks. If necessary, they are good basis for further performance tuning.

* `bert`: a transformer classifier to fine-tune, to be instanciated by any BERT pre-trained model or transformers available on HuggingFace Hub (we have tested various BERT and RoBERTa flavors) 
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

Note: by default the first 300 tokens of the text to be classified are used, which is largely enough for any _short text_ classification tasks and works fine with low profile GPU (for instance GeForce GTX 1050 Ti with 4 GB memory). For taking into account a larger portion of the text, modify the config model parameter `maxlen`. However, using more than 1000 tokens for instance requires a modern GPU with enough memory (e.g. 10 GB).

