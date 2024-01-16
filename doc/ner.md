# DeLFT NER applications

See [here](https://delft.readthedocs.io/en/latest/sequence_labeling/) for the list of supported sequence labeling architectures. 

In general, the best results will be obtained with `BidLSTM_CRF` architecture together with `ELMo` (ELMo is particularly good for sequence labeling, so don't forget ELMo!) or with `BERT_CRF` using a pretrained transformer model specialized in the NER domain of the application (e.g. SciBERT for scientific NER, CamemBERT for general NER on French, etc.).

NER models can be trained and applied via the script `delft/applications/nerTagger.py`. We describe on this page some available models and results obtained with DeLFT applied to standard NER tasks. 

See [NER Datasets](ner-datasets.md) for more information on the datasets used in this section. 

## Overview

We have reimplemented in DeLFT some reference neural architectures for NER of the last four years and performed a reproducibility analysis of the these systems with comparable evaluation criterias. Unfortunaltely, in publications, systems are usually compared directly with reported results obtained in different settings, which can bias scores by more than 1.0 point and completely invalidate both comparison and interpretation of results.  

You can read more about our reproducibility study of neural NER in this [blog article](http://science-miner.com/a-reproducibility-study-on-neural-ner/). This effort is similar to the work of [(Yang and Zhang, 2018)](https://arxiv.org/pdf/1806.04470.pdf) (see also [NCRFpp](https://github.com/jiesutd/NCRFpp)) but has also been extended to BERT for a fair comparison of RNN for sequence labeling, and can also be related to the motivations of [(Pressel et al., 2018)](http://aclweb.org/anthology/W18-2506) [MEAD](https://github.com/dpressel/mead-baseline). 

All reported scores bellow are __f-score__ for the CoNLL-2003 NER dataset. We report first the f-score averaged over 10 training runs, and second the best f-score over these 10 training runs. All the DeLFT trained models are included in this repository. 

| Architecture  | Implementation | Glove only (avg / best)| Glove + valid. set (avg / best)| ELMo + Glove (avg / best)| ELMo + Glove + valid. set (avg / best)|
| --- | --- | --- | --- | --- | --- |
| BidLSTM_CRF   | DeLFT | __91.03__ / __91.38__  | 91.37 / 91.69 | __92.57__ / __92.80__ | __92.95__ / __93.21__ | 
|               | [(Lample and al., 2016)](https://arxiv.org/abs/1603.01360) | - / 90.94 |      |              |               | 
| BidLSTM_CNN_CRF | DeLFT | 90.64 / 91.23| 90.98 / 91.38 | 92.30 / 92.57| 92.67 / 93.04 |
|               | [(Ma & Hovy, 2016)](https://arxiv.org/abs/1603.01354) |  - / 91.21  | | | |
|               | [(Peters & al. 2018)](https://arxiv.org/abs/1802.05365) |  | | 92.22** / - | |
| BidLSTM_CNN   | DeLFT | 89.49 / 89.96  | 89.85 / 90.13 | 91.66 / 92.00 | 92.01 / 92.16 |
|               | [(Chiu & Nichols, 2016)](https://arxiv.org/abs/1511.08308) || __90.88***__ / - | | |
| BidGRU_CRF    | DeLFT | 90.17 / 90.55  | 91.04 / 91.40 | 92.03 / 92.44 | 92.43 / 92.71 |
|               | [(Peters & al. 2017)](https://arxiv.org/abs/1705.00108) |  | |  | 91.93* / - |

Results with transformer fine-tuning for CoNLL-2003 NER dataset, including a final CRF activation layer, instead of a softmax. A CRF activation layer improves f-score in average by around +0.10 for sequence labelling task, but increase the runtime by 23%: 

| Architecture  | pretrained model | Implementation | f-score |
| ---  | --- | --- | --- |
| BERT | bert-base-cased     | DeLFT | 91.19 |  
| BERT_CRF | bert-base-cased +CRF| DeLFT | 91.25 |  
| BERT_ChainCRF | bert-base-cased +CRF| DeLFT | 91.22 |  
| BERT | roberta-base     | DeLFT | 91.64 |  

Note: DeLFT uses `BERT` as architecture name for transformers in general, but the transformer model could be in principle any transformer variants preset in HuggingFace Hub. DeLFT supports 2 implementations of a CRF layer to be combined with RNN and transformer architectures: `CRF` based on TensorFlow Addons and `ChainCRF` a custom implementation. Both should produce similar accuracy results, but `ChainCRF` is significantly faster and robust. 

For reference, the original reported result for  `bert-base-cased` model in [(Devlin & al. 2018)](https://arxiv.org/abs/1810.04805) is **92.4**, using "document context".

For DeLFT, the average is obtained with 10 training runs (see latest [full results](https://github.com/kermitt2/delft/blob/master/doc/sequence_labeling.0.3.0.txt)) and for (Devlin & al. 2018) averaged with 5 runs. As noted [here](https://github.com/google-research/bert/issues/223), the original CoNLL-2003 NER results with BERT reported by the Google Research paper are not easily reproducible, and the score obtained by DeLFT is very similar to those obtained by all the systems having reproduced this experiment in similar condition (e.g. without "document context"). 

_*_ reported f-score using Senna word embeddings and not Glove.

** f-score is averaged over 5 training runs. 

*** reported f-score with Senna word embeddings (Collobert 50d) averaged over 10 runs, including case features and not including lexical features. DeLFT implementation of the same architecture includes the capitalization features too, but uses the more efficient GloVe 300d embeddings.


##### Command Line Interface

Different datasets and languages are supported. They can be specified by the command line parameters. The general usage of the CLI is as follow: 

```
usage: nerTagger.py [-h] [--fold-count FOLD_COUNT] [--lang LANG] [--dataset-type DATASET_TYPE]
                    [--train-with-validation-set] [--architecture ARCHITECTURE] [--data-path DATA_PATH]
                    [--file-in FILE_IN] [--file-out FILE_OUT] [--embedding EMBEDDING]
                    [--transformer TRANSFORMER]
                    action

Neural Named Entity Recognizers based on DeLFT

positional arguments:
  action                one of [train, train_eval, eval, tag]

optional arguments:
  -h, --help            show this help message and exit
  --fold-count FOLD_COUNT
                        number of folds or re-runs to be used when training
  --lang LANG           language of the model as ISO 639-1 code (en, fr, de, etc.)
  --dataset-type DATASET_TYPE
                        dataset to be used for training the model
  --train-with-validation-set
                        Use the validation set for training together with the training set
  --architecture ARCHITECTURE
                        type of model architecture to be used, one of ['BidLSTM_CRF', 'BidLSTM_CNN_CRF',
                        'BidLSTM_CNN_CRF', 'BidGRU_CRF', 'BidLSTM_CNN', 'BidLSTM_CRF_CASING', 'BERT',
                        'BERT_CRF', 'BERT_CRF_FEATURES', 'BERT_CRF_CHAR', 'BERT_CRF_CHAR_FEATURES']
  --data-path DATA_PATH
                        path to the corpus of documents for training (only use currently with Ontonotes
                        corpus in orginal XML format)
  --file-in FILE_IN     path to a text file to annotate
  --file-out FILE_OUT   path for outputting the resulting JSON NER anotations
  --embedding EMBEDDING
                        The desired pre-trained word embeddings using their descriptions in the file. For
                        local loading, use delft/resources-registry.json. Be sure to use here the same
                        name as in the registry, e.g. ['glove-840B', 'fasttext-crawl', 'word2vec'] and
                        that the path in the registry to the embedding file is correct on your system.
  --transformer TRANSFORMER
                        The desired pre-trained transformer to be used in the selected architecture. For
                        local loading use, delft/resources-registry.json, and be sure to use here the
                        same name as in the registry, e.g. ['bert-base-cased', 'bert-large-cased',
                        'allenai/scibert_scivocab_cased'] and that the path in the registry to the model
                        path is correct on your system. HuggingFace transformers hub will be used
                        otherwise to fetch the model, see https://huggingface.co/models for model names
```


## CONLL 2003

DeLFT comes with various trained models for the CoNLL-2003 NER dataset.

By default, the `BidLSTM_CRF` architecture is used.  

Using `BidLSTM_CRF` model with ELMo embeddings, following [7] and some parameter optimisations and [warm-up](https://github.com/allenai/allennlp/blob/master/docs/tutorials/how_to/elmo.md#notes-on-statefulness-and-non-determinism), improve the f1 score on CoNLL 2003 significantly.

For re-training a model, the CoNLL-2003 NER dataset (`eng.train`, `eng.testa`, `eng.testb`) must be present under `data/sequenceLabelling/CoNLL-2003/` in IOB2 tagging sceheme (look [here](https://github.com/Franck-Dernoncourt/NeuroNER/tree/4cbfc3a1b4c4a5242e1cfbaea48d6f7e972e8881/data/conll2003/en) for instance ;) and [here](https://github.com/kermitt2/delft/tree/master/delft/utilities). The CONLL 2003 dataset (English) is the default dataset and English is the default language, but you can also indicate it explicitly as parameter with `--dataset-type conll2003` and specifying explicitly the language `--lang en`.

For training and evaluating following the traditional approach (training with the train set without validation set, and evaluating on test set), use:

```sh
> python3 delft/applications/nerTagger.py --dataset-type conll2003 train_eval
```

Some recent works like (Chiu & Nichols, 2016) and (Peters and al., 2017) also train with the validation set, leading obviously to a better accuracy (still they compare their scores with scores previously reported trained differently, which is arguably a bit unfair - this aspect is mentioned in (Ma & Hovy, 2016)). To train with both train and validation sets, use the parameter `--train-with-validation-set`:

> python3 delft/applications/nerTagger.py --dataset-type conll2003 --train-with-validation-set train_eval

Note that, by default, the `BidLSTM_CRF` model is used. (Documentation on selecting other models and setting hyperparameters to be included here !)

For evaluating against CoNLL 2003 testb set with the existing model:

```sh
> python3 delft/applications/nerTagger.py --dataset-type conll2003 eval
```

```text
    Evaluation on test set:
        f1 (micro): 91.35
                 precision    recall  f1-score   support

            ORG     0.8795    0.9007    0.8899      1661
            PER     0.9647    0.9623    0.9635      1617
           MISC     0.8261    0.8120    0.8190       702
            LOC     0.9260    0.9305    0.9282      1668

    avg / total     0.9109    0.9161    0.9135      5648

```

If the model has been trained also with the validation set (`--train-with-validation-set`), similarly to (Chiu & Nichols, 2016) or (Peters and al., 2017), results are significantly better:

```text
    Evaluation on test set:
        f1 (micro): 91.60
                 precision    recall  f1-score   support

            LOC     0.9219    0.9418    0.9318      1668
           MISC     0.8277    0.8077    0.8176       702
            PER     0.9594    0.9635    0.9614      1617
            ORG     0.9029    0.8904    0.8966      1661

    avg / total     0.9158    0.9163    0.9160      5648
```

Using ELMo with the best model obtained over 10 training (not using the validation set for training, only for early stop):

```text
    Evaluation on test set:
        f1 (micro): 92.80
                  precision    recall  f1-score   support

             LOC     0.9401    0.9412    0.9407      1668
            MISC     0.8104    0.8405    0.8252       702
             ORG     0.9107    0.9151    0.9129      1661
             PER     0.9800    0.9722    0.9761      1617

all (micro avg.)     0.9261    0.9299    0.9280      5648
```

Using BERT architecture for sequence labelling (pre-trained transformer with fine-tuning), for instance here the `bert-base-cased`, cased, pre-trained model, use:

```sh
> python3 delft/applications/nerTagger.py --architecture BERT_CRF --dataset-type conll2003 --fold-count 10 --transformer bert-base-cased train_eval
```

```text
average over 10 folds
            precision    recall  f1-score   support

       ORG     0.8804    0.9114    0.8957      1661
      MISC     0.7823    0.8189    0.8002       702
       PER     0.9633    0.9576    0.9605      1617
       LOC     0.9290    0.9316    0.9303      1668

  macro f1 = 0.9120
  macro precision = 0.9050
  macro recall = 0.9191

```

For training with all the available data:

```sh
> python3 delft/applications/nerTagger.py --dataset-type conll2003 train
```

To take into account the strong impact of random seed, you need to train multiple times with the n-folds options. The model will be trained n times with different seed values but with the same sets if the evaluation set is provided. The evaluation will then give the average scores over these n models (against test set) and for the best model which will be saved. For 10 times training for instance, use:

```sh
> python3 delft/applications/nerTagger.py --dataset-type conll2003 --fold-count 10 train_eval
```

After training a model, for tagging some text, for instance in a file `data/test/test.ner.en.txt` (), use the command:

```sh
> python3 delft/applications/nerTagger.py --dataset-type conll2003 --file-in data/test/test.ner.en.txt tag
```

For instance for tagging the text with a specific architecture that has been previously trained: 

```sh
> python3 delft/applications/nerTagger.py --dataset-type conll2003 --file-in data/test/test.ner.en.txt --architecture BERT_CRF_FEATURES --transformer bert-base-cased tag
```

Note that, currently, the input text file must contain one sentence per line, so the text must be presegmented into sentences. To obtain the JSON annotations in a text file instead than in the standard output, use the parameter `--file-out`. Predictions work at around 7400 tokens per second for the BidLSTM_CRF architecture with a GeForce GTX 1080 Ti. 

This produces a JSON output with entities, scores and character offsets like this:

```json
{
    "runtime": 0.34,
    "texts": [
        {
            "text": "The University of California has found that 40 percent of its students suffer food insecurity. At four state universities in Illinois, that number is 35 percent.",
            "entities": [
                {
                    "text": "University of California",
                    "endOffset": 32,
                    "score": 1.0,
                    "class": "ORG",
                    "beginOffset": 4
                },
                {
                    "text": "Illinois",
                    "endOffset": 134,
                    "score": 1.0,
                    "class": "LOC",
                    "beginOffset": 125
                }
            ]
        },
        {
            "text": "President Obama is not speaking anymore from the White House.",
            "entities": [
                {
                    "text": "Obama",
                    "endOffset": 18,
                    "score": 1.0,
                    "class": "PER",
                    "beginOffset": 10
                },
                {
                    "text": "White House",
                    "endOffset": 61,
                    "score": 1.0,
                    "class": "LOC",
                    "beginOffset": 49
                }
            ]
        }
    ],
    "software": "DeLFT",
    "date": "2018-05-02T12:24:55.529301",
    "model": "ner"
}

```

For English NER tagging, when used, the default static embeddings is Glove (`glove-840B`). Other static embeddings can be specified with the parameter `--embedding`, for instance:

```sh
> python3 delft/applications/nerTagger.py --dataset-type conll2003 --embedding word2vec train_eval
```

## Ontonotes 5.0 CONLL 2012

DeLFT comes with pre-trained models with the [Ontonotes 5.0 CoNLL-2012 NER dataset](http://cemantix.org/data/ontonotes.html). As dataset-type identifier, use `conll2012`. All the options valid for CoNLL-2003 NER dataset are usable for this dataset. Static embeddings for Ontonotes can be set with parameter `--embedding`.

For re-training, the assembled Ontonotes datasets following CoNLL-2012 must be available and converted into IOB2 tagging scheme, see [here](https://github.com/kermitt2/delft/tree/master/delft/utilities) for more details. To train and evaluate following the traditional approach (training with the train set without validation set, and evaluating on test set), with `BidLSTM_CRF` architecture use:

```sh
> python3 nerTagger.py --dataset-type conll2012 train_eval --architecture BidLSTM_CRF --embedding glove-840B
```

```text
training runtime: 23692.0 seconds

Evaluation on test set:

    f1 (micro): 87.01
                  precision    recall  f1-score   support

            DATE     0.8029    0.8695    0.8349      1602
        CARDINAL     0.8130    0.8139    0.8135       935
          PERSON     0.9061    0.9371    0.9214      1988
             GPE     0.9617    0.9411    0.9513      2240
             ORG     0.8799    0.8568    0.8682      1795
           MONEY     0.8903    0.8790    0.8846       314
            NORP     0.9226    0.9501    0.9361       841
         ORDINAL     0.7873    0.8923    0.8365       195
            TIME     0.5772    0.6698    0.6201       212
     WORK_OF_ART     0.6000    0.5060    0.5490       166
             LOC     0.7340    0.7709    0.7520       179
           EVENT     0.5000    0.5556    0.5263        63
         PRODUCT     0.6528    0.6184    0.6351        76
         PERCENT     0.8717    0.8567    0.8642       349
        QUANTITY     0.7155    0.7905    0.7511       105
             FAC     0.7167    0.6370    0.6745       135
        LANGUAGE     0.8462    0.5000    0.6286        22
             LAW     0.7308    0.4750    0.5758        40

all (micro avg.)     0.8647    0.8755    0.8701     11257
```

With `bert-base-cased` `BERT_CRF` architecture:

```sh
> python3 delft/applications/nerTagger.py train_eval --dataset-type conll2012 --architecture BERT_CRF --transformer bert-base-cased
```

```
training runtime: 14367.8 seconds

Evaluation on test set:

                  precision    recall  f1-score   support

        CARDINAL     0.8443    0.8064    0.8249       935
            DATE     0.8474    0.8770    0.8620      1602
           EVENT     0.7460    0.7460    0.7460        63
             FAC     0.7163    0.7481    0.7319       135
             GPE     0.9657    0.9437    0.9546      2240
        LANGUAGE     0.8889    0.7273    0.8000        22
             LAW     0.6857    0.6000    0.6400        40
             LOC     0.6965    0.7821    0.7368       179
           MONEY     0.8882    0.9108    0.8994       314
            NORP     0.9350    0.9584    0.9466       841
         ORDINAL     0.8199    0.8872    0.8522       195
             ORG     0.8908    0.8997    0.8952      1795
         PERCENT     0.8917    0.8968    0.8943       349
          PERSON     0.9396    0.9472    0.9434      1988
         PRODUCT     0.5600    0.7368    0.6364        76
        QUANTITY     0.6187    0.8190    0.7049       105
            TIME     0.6184    0.6651    0.6409       212
     WORK_OF_ART     0.6138    0.6988    0.6535       166

all (micro avg.)     0.8825    0.8951    0.8888     11257
```

With ELMo embeddings (using the default hyper-parameters, except the batch size which is increased to better learn the less frequent classes):

```sh
> python3 delft/applications/nerTagger.py train_eval --dataset-type conll2012 --architecture BidLSTM_CRF --embedding glove-840B --use-ELMo
```

```
training runtime: 36812.025 seconds 

Evaluation on test set:
                  precision    recall  f1-score   support

        CARDINAL     0.8534    0.8342    0.8437       935
            DATE     0.8499    0.8733    0.8615      1602
           EVENT     0.7091    0.6190    0.6610        63
             FAC     0.7667    0.6815    0.7216       135
             GPE     0.9682    0.9527    0.9604      2240
        LANGUAGE     0.9286    0.5909    0.7222        22
             LAW     0.7000    0.5250    0.6000        40
             LOC     0.7759    0.7542    0.7649       179
           MONEY     0.9054    0.9140    0.9097       314
            NORP     0.9323    0.9501    0.9411       841
         ORDINAL     0.8082    0.9077    0.8551       195
             ORG     0.8950    0.9019    0.8984      1795
         PERCENT     0.9117    0.9169    0.9143       349
          PERSON     0.9430    0.9482    0.9456      1988
         PRODUCT     0.6410    0.6579    0.6494        76
        QUANTITY     0.7890    0.8190    0.8037       105
            TIME     0.6683    0.6462    0.6571       212
     WORK_OF_ART     0.6301    0.6566    0.6431       166

all (micro avg.)     0.8943    0.8956    0.8949     11257
```

## French model (based on Le Monde corpus)

Note that Le Monde corpus is subject to copyrights and is limited to research usage only, it is usually referred to as "corpus FTB". The corpus file `ftb6_ALL.EN.docs.relinked.xml` must be located under `delft/data/sequenceLabelling/leMonde/`. This is the default French model, so it will be used by simply indicating the language as parameter: `--lang fr`, but you can also indicate explicitly the dataset with `--dataset-type ftb`. Default static embeddings for French language models are `wiki.fr`, which can be changed with parameter `--embedding`.

Similarly as before, for training and evaluating use:

```sh
> python3 delft/applications/nerTagger.py --lang fr --dataset-type ftb train_eval
```

In practice, we need to repeat training and evaluation several times to neutralise random seed effects and to average scores, here ten times:

```sh
> python3 delft/applications/nerTagger.py --lang fr --dataset-type ftb --fold-count 10 train_eval
```

The performance is as follow, for the `BiLSTM_CRF` architecture and fasttext `wiki.fr` embeddings, averaged over 10 training:

```text
----------------------------------------------------------------------

** Worst ** model scores - run 2
                  precision    recall  f1-score   support

      <artifact>     1.0000    0.5000    0.6667         8
      <business>     0.8242    0.8772    0.8499       342
   <institution>     0.8571    0.7826    0.8182        23
      <location>     0.9386    0.9582    0.9483       383
  <organisation>     0.8750    0.7292    0.7955       240
        <person>     0.9631    0.9457    0.9543       221

all (micro avg.)     0.8964    0.8817    0.8890      1217


** Best ** model scores - run 3
                  precision    recall  f1-score   support

      <artifact>     1.0000    0.7500    0.8571         8
      <business>     0.8457    0.8977    0.8709       342
   <institution>     0.8182    0.7826    0.8000        23
      <location>     0.9367    0.9661    0.9512       383
  <organisation>     0.8832    0.7875    0.8326       240
        <person>     0.9459    0.9502    0.9481       221

all (micro avg.)     0.9002    0.9039    0.9020      1217

----------------------------------------------------------------------

Average over 10 folds
                  precision    recall  f1-score   support

      <artifact>     1.0000    0.6000    0.7432         8
      <business>     0.8391    0.8830    0.8605       342
   <institution>     0.8469    0.7652    0.8035        23
      <location>     0.9388    0.9645    0.9514       383
  <organisation>     0.8644    0.7592    0.8079       240
        <person>     0.9463    0.9529    0.9495       221

all (micro avg.)     0.8961    0.8929    0.8945
```

With frELMo:

```text
----------------------------------------------------------------------

** Worst ** model scores - run 2
                  precision    recall  f1-score   support

      <artifact>     1.0000    0.5000    0.6667         8
      <business>     0.8704    0.9035    0.8867       342
   <institution>     0.8000    0.6957    0.7442        23
      <location>     0.9342    0.9634    0.9486       383
  <organisation>     0.8043    0.7875    0.7958       240
        <person>     0.9641    0.9729    0.9685       221

all (micro avg.)     0.8945    0.9055    0.9000      1217


** Best ** model scores - run 3
                  precision    recall  f1-score   support

      <artifact>     1.0000    0.7500    0.8571         8
      <business>     0.8883    0.9298    0.9086       342
   <institution>     0.8500    0.7391    0.7907        23
      <location>     0.9514    0.9713    0.9612       383
  <organisation>     0.8597    0.7917    0.8243       240
        <person>     0.9774    0.9774    0.9774       221

all (micro avg.)     0.9195    0.9195    0.9195      1217

----------------------------------------------------------------------

Average over 10 folds
                  precision    recall  f1-score   support

      <artifact>     0.8833    0.5125    0.6425         8
      <business>     0.8803    0.9067    0.8933       342
   <institution>     0.7933    0.7391    0.7640        23
      <location>     0.9438    0.9679    0.9557       383
  <organisation>     0.8359    0.8004    0.8176       240
        <person>     0.9699    0.9760    0.9729       221

all (micro avg.)     0.9073    0.9118    0.9096 
```

Using `camembert-base` as transformer layer in a `BERT_CRF` architecture: 

```sh
> python3 delft/applications/nerTagger.py --lang fr --dataset-type ftb train_eval --architecture BERT_CRF --transformer camembert-base
```

```
                  precision    recall  f1-score   support

      <artifact>     0.0000    0.0000    0.0000         8
      <business>     0.8940    0.9123    0.9030       342
   <institution>     0.6923    0.7826    0.7347        23
      <location>     0.9563    0.9713    0.9637       383
  <organisation>     0.8270    0.8167    0.8218       240
        <person>     0.9688    0.9819    0.9753       221

all (micro avg.)     0.9102    0.9162    0.9132      1217
```

For historical reason, we can also consider a particular split of the FTB corpus into train, dev and set set and with a forced tokenization (like the old CoNLL 2013 NER), that was used in previous work for comparison. Obviously the evaluation is dependent to this particular set and the n-fold cross validation is a much better practice and should be prefered (as well as a format that do not force a tokenization). For using the forced split FTB (using the files `ftb6_dev.conll`, `ftb6_test.conll` and `ftb6_train.conll` located under `delft/data/sequenceLabelling/leMonde/`), use as parameter `--dataset-type ftb_force_split`:

```sh
> python3 delft/applications/nerTagger.py --lang fr --dataset-type ftb_force_split --fold-count 10 train_eval
```

which gives for the BiLSTM-CRF architecture and fasttext `wiki.fr` embeddings averaged over 10 training:

```
----------------------------------------------------------------------

** Worst ** model scores - run 4
                  precision    recall  f1-score   support

         Company     0.7908    0.7690    0.7797       290
FictionCharacter     0.0000    0.0000    0.0000         2
        Location     0.9164    0.9164    0.9164       347
    Organization     0.7895    0.7235    0.7550       311
          Person     0.9000    0.9220    0.9108       205
         Product     1.0000    0.3333    0.5000         3

all (micro avg.)     0.8498    0.8256    0.8375      1158


** Best ** model scores - run 0
                  precision    recall  f1-score   support

         Company     0.8026    0.8552    0.8280       290
FictionCharacter     0.0000    0.0000    0.0000         2
        Location     0.9326    0.9164    0.9244       347
    Organization     0.8244    0.7395    0.7797       311
          Person     0.8826    0.9171    0.8995       205
         Product     1.0000    1.0000    1.0000         3

all (micro avg.)     0.8620    0.8523    0.8571      1158

----------------------------------------------------------------------

Average over 10 folds
                  precision    recall  f1-score   support

         Company     0.7920    0.8148    0.8030       290
FictionCharacter     0.0000    0.0000    0.0000         2
        Location     0.9234    0.9098    0.9165       347
    Organization     0.8071    0.7328    0.7681       311
             POI     0.0000    0.0000    0.0000         0
          Person     0.8974    0.9254    0.9112       205
         Product     1.0000    0.9000    0.9300         3

all (micro avg.)     0.8553    0.8396    0.8474  
```

With frELMo:

```
----------------------------------------------------------------------

** Worst ** model scores - run 3
                  precision    recall  f1-score   support

         Company     0.8215    0.8414    0.8313       290
FictionCharacter     0.0000    0.0000    0.0000         2
        Location     0.9020    0.9280    0.9148       347
    Organization     0.7833    0.7556    0.7692       311
          Person     0.9327    0.9463    0.9395       205
         Product     0.0000    0.0000    0.0000         3

all (micro avg.)     0.8563    0.8592    0.8578      1158


** Best ** model scores - run 1
                  precision    recall  f1-score   support

         Company     0.8289    0.8690    0.8485       290
FictionCharacter     0.0000    0.0000    0.0000         2
        Location     0.9290    0.9424    0.9356       347
    Organization     0.8475    0.7685    0.8061       311
          Person     0.9327    0.9463    0.9395       205
         Product     0.6667    0.6667    0.6667         3

all (micro avg.)     0.8825    0.8756    0.8791      1158

----------------------------------------------------------------------

Average over 10 folds
                  precision    recall  f1-score   support

         Company     0.8195    0.8503    0.8346       290
FictionCharacter     0.0000    0.0000    0.0000         2
        Location     0.9205    0.9363    0.9283       347
    Organization     0.8256    0.7595    0.7910       311
             POI     0.0000    0.0000    0.0000         0
          Person     0.9286    0.9454    0.9369       205
         Product     0.7417    0.6667    0.6824         3

all (micro avg.)     0.8718    0.8666    0.8691
```

For the `ftb_force_split` dataset, similarly as for CoNLL 2013, you can use the `train_with_validation_set` parameter to add the validation set in the training data. The above results are all obtained without using `train_with_validation_set` (which is the common approach).

Finally, for training with all the dataset without evaluation (e.g. for production):

```sh
> python3 delft/applications/nerTagger.py --lang fr --dataset-type ftb train
```

and for annotating some examples:

```sh
> python3 delft/applications/nerTagger.py --lang fr --dataset-type ftb --file-in data/test/test.ner.fr.txt tag
```

```json
{
    "date": "2018-06-11T21:25:03.321818",
    "runtime": 0.511,
    "software": "DeLFT",
    "model": "ner-fr-lemonde",
    "texts": [
        {
            "entities": [
                {
                    "beginOffset": 5,
                    "endOffset": 13,
                    "score": 1.0,
                    "text": "Allemagne",
                    "class": "<location>"
                },
                {
                    "beginOffset": 57,
                    "endOffset": 68,
                    "score": 1.0,
                    "text": "Donald Trump",
                    "class": "<person>"
                }
            ],
            "text": "Or l’Allemagne pourrait préférer la retenue, de peur que Donald Trump ne surtaxe prochainement les automobiles étrangères."
        }
    ]
}

```

<p align="center">
    <img src="https://abstrusegoose.com/strips/muggle_problems.png">
</p>

This above work is licensed under a [Creative Commons Attribution-Noncommercial 3.0 United States License](http://creativecommons.org/licenses/by-nc/3.0/us/). 


## Insult recognition

A small experimental model for recognising insults and threats in texts, based on the Wikipedia comment from the Kaggle _Wikipedia Toxic Comments_ dataset, English only. This uses a small dataset labelled manually.

```
usage: insultTagger.py [-h] [--fold-count FOLD_COUNT] [--architecture ARCHITECTURE]
                       [--embedding EMBEDDING] [--transformer TRANSFORMER]
                       action

Experimental insult recognizer for the Wikipedia toxic comments dataset

positional arguments:
  action

optional arguments:
  -h, --help            show this help message and exit
  --fold-count FOLD_COUNT
  --architecture ARCHITECTURE
                        Type of model architecture to be used, one of ['BidLSTM_CRF', 'BidLSTM_CNN_CRF',
                        'BidLSTM_CNN_CRF', 'BidGRU_CRF', 'BidLSTM_CNN', 'BidLSTM_CRF_CASING', 'BERT',
                        'BERT_CRF', 'BERT_CRF_FEATURES', 'BERT_CRF_CHAR', 'BERT_CRF_CHAR_FEATURES']
  --embedding EMBEDDING
                        The desired pre-trained word embeddings using their descriptions in the file. For
                        local loading, use delft/resources-registry.json. Be sure to use here the same
                        name as in the registry, e.g. ['glove-840B', 'fasttext-crawl', 'word2vec'] and
                        that the path in the registry to the embedding file is correct on your system.
  --transformer TRANSFORMER
                        The desired pre-trained transformer to be used in the selected architecture. For
                        local loading use, delft/resources-registry.json, and be sure to use here the
                        same name as in the registry, e.g. ['bert-base-cased', 'bert-large-cased',
                        'allenai/scibert_scivocab_cased'] and that the path in the registry to the model
                        path is correct on your system. HuggingFace transformers hub will be used
                        otherwise to fetch the model, see https://huggingface.co/models for model names
```

For training:

```sh
> python3 delft/applications/insultTagger.py train
```

By default training uses the whole train set.

Example of a small tagging test:

```sh
> python3 delft/applications/insultTagger.py tag
```

will produced (__socially offensive language warning!__) result like this:

```json
{
    "runtime": 0.969,
    "texts": [
        {
            "entities": [],
            "text": "This is a gentle test."
        },
        {
            "entities": [
                {
                    "score": 1.0,
                    "endOffset": 20,
                    "class": "<insult>",
                    "beginOffset": 9,
                    "text": "moronic wimp"
                },
                {
                    "score": 1.0,
                    "endOffset": 56,
                    "class": "<threat>",
                    "beginOffset": 54,
                    "text": "die"
                }
            ],
            "text": "you're a moronic wimp who is too lazy to do research! die in hell !!"
        }
    ],
    "software": "DeLFT",
    "date": "2018-05-14T17:22:01.804050",
    "model": "insult"
}
```
