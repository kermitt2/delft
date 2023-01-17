# Transformers layers

## Loading sources 

| Transformer                    | description                               | registry?                                                   | huggingface? | Loading method                                   |  
|--------------------------------|-------------------------------------------|-------------------------------------------------------------|--------------|--------------------------------------------------|
| allenai/scibert_cased_scivocab | SciBERT from HuggingFace                  | no                                                          | yes          | AutoModel/AutoTokenizer via Hugging face calls   | 
| portiz/matbert                 | RoBERTa-based trained PyTorch transformer | yes, only the local directory is needed                     | no           | AutoModel/AutoTokenizer via local directory load | 
| scibert                        | SciBERT vanilla from GitHub               | yes, weights, config and tokenizer are specified in the log | no           | ???                                              |


Examples of configuration: 

 - **portiz/matbert** as a locally saved pretrained model:

    ```json
            {
                "name": "portiz/matbert",
                "model_dir": "/Users/lfoppiano/development/projects/embeddings/pre-trained-embeddings/matbert",
                "lang": "en"
            }
    ```
