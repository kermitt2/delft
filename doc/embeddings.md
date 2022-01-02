# Management of embeddings

The first time DeLFT starts and accesses pre-trained embeddings, these embeddings are serialised and stored in a LMDB database, a very efficient embedded database using memory-mapped file (already used in the Machine Learning world by Caffe and Torch for managing large training data). The next time these embeddings will be accessed, they will be immediately available.

Our approach solves the bottleneck problem pointed for instance [here](https://spenai.org/bravepineapple/faster_em/) in a much better way than quantising+compression or pruning. After being compiled and stored at the first access, any volume of embeddings vectors can be used immediately without any loading, with a negligible usage of memory, without any accuracy loss and with a negligible impact on runtime when using SSD. In practice, we can exploit for instance embeddings for dozen languages simultaneously, without any memory and runtime issues - a requirement for any ambitious industrial deployment of a neural NLP system. 

For instance, in a traditional approach `glove-840B` takes around 2 minutes to load and 4GB in memory. Managed with LMDB, after a first load time of around 4 minutes, `glove-840B` can be accessed immediately and takes only a couple MB in memory, for an impact on runtime negligible (around 1% slower) for any further command line calls.

By default, the LMDB databases are stored under the subdirectory `data/db`. The size of a database is roughly equivalent to the size of the original uncompressed embeddings file. To modify this path, edit the file `delft/resources-registry.json` and change the value of the attribute `embedding-lmdb-path`.

> I have plenty of memory on my machine, I don't care about load time because I need to grab a coffee every ten minutes, I only process one language at the time, so I am not interested in taking advantage of the LMDB emebedding management !

Ok, ok, then set the `embedding-lmdb-path` value to `"None"` in the file `delft/resources-registry.json`, the embeddings will be loaded in memory as immutable data, like in the usual Keras scripts.

