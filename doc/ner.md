# DeLFT NER applications

NER models can be trained and applied via the script `delft/applications/nerTagger.py`. 

See [NER Datasets](ner-datasets.md) for more information on the datasets used in this section. 

## Overview

We have reimplemented in DeLFT some reference neural architectures for NER of the last four years and performed a reproducibility analysis of the these systems with comparable evaluation criterias. Unfortunaltely, in publications, systems are usually compared directly with reported results obtained in different settings, which can bias scores by more than 1.0 point and completely invalidate both comparison and interpretation of results.  

You can read more about our reproducibility study of neural NER in this [blog article](http://science-miner.com/a-reproducibility-study-on-neural-ner/). This effort is similar to the work of [(Yang and Zhang, 2018)](https://arxiv.org/pdf/1806.04470.pdf) (see also [NCRFpp](https://github.com/jiesutd/NCRFpp)) but has also been extended to BERT for a fair comparison of RNN for sequence labeling, and can also be related to the motivations of [(Pressel et al., 2018)](http://aclweb.org/anthology/W18-2506) [MEAD](https://github.com/dpressel/mead-baseline). 

All reported scores bellow are __f-score__ for the CoNLL-2003 NER dataset. We report first the f-score averaged over 10 training runs, and second the best f-score over these 10 training runs. All the DeLFT trained models are included in this repository. 

| Architecture  | Implementation | Glove only (avg / best)| Glove + valid. set (avg / best)| ELMo + Glove (avg / best)| ELMo + Glove + valid. set (avg / best)|
| --- | --- | --- | --- | --- | --- |
| BidLSTM-CRF   | DeLFT | __91.03__ / __91.38__  | 91.37 / 91.69 | __92.57__ / __92.80__ | __92.95__ / __93.21__ | 
|               | [(Lample and al., 2016)](https://arxiv.org/abs/1603.01360) | - / 90.94 |      |              |               | 
| BidLSTM-CNN-CRF | DeLFT | 90.64 / 91.23| 90.98 / 91.38 | 92.30 / 92.57| 92.67 / 93.04 |
|               | [(Ma & Hovy, 2016)](https://arxiv.org/abs/1603.01354) |  - / 91.21  | | | |
|               | [(Peters & al. 2018)](https://arxiv.org/abs/1802.05365) |  | | 92.22** / - | |
| BidLSTM-CNN   | DeLFT | 89.49 / 89.96  | 89.85 / 90.13 | 91.66 / 92.00 | 92.01 / 92.16 |
|               | [(Chiu & Nichols, 2016)](https://arxiv.org/abs/1511.08308) || __90.88***__ / - | | |
| BidGRU-CRF    | DeLFT | 90.17 / 90.55  | 91.04 / 91.40 | 92.03 / 92.44 | 92.43 / 92.71 |
|               | [(Peters & al. 2017)](https://arxiv.org/abs/1705.00108) |  | |  | 91.93* / - |

Results with BERT fine-tuning for CoNLL-2003 NER dataset, including a final CRF activation layer, instead of a softmax. A CRF activation layer improves f-score in average by around +0.10 for sequence labelling task, but increase the runtime by 23%: 

| Architecture  | Implementation | f-score |
| --- | --- | --- | 
| bert-base-en    | DeLFT | 91.19 |  
| bert-base-en+CRF    | DeLFT | 91.25 |  
| bert-base-en        | [(Devlin & al. 2018)](https://arxiv.org/abs/1810.04805) | 92.4 |

For DeLFT, the average is obtained with 10 training runs (see latest [full results](https://github.com/kermitt2/delft/blob/master/doc/sequence_labeling.0.3.0.txt)) and for (Devlin & al. 2018) averaged with 5 runs. As noted [here](https://github.com/google-research/bert/issues/223), the original CoNLL-2003 NER results with BERT reported by the Google Research paper are not easily reproducible (if reproducible), and the score obtained by DeLFT is very similar to those obtained by all the systems having reproduced this experiment in similar condition. 

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

By default, the BidLSTM-CRF architecture is used.  

Using BidLSTM-CRF model with ELMo embeddings, following [7] and some parameter optimisations and [warm-up](https://github.com/allenai/allennlp/blob/master/docs/tutorials/how_to/elmo.md#notes-on-statefulness-and-non-determinism), improve the f1 score on CoNLL 2003 significantly.

For re-training a model, the CoNLL-2003 NER dataset (`eng.train`, `eng.testa`, `eng.testb`) must be present under `data/sequenceLabelling/CoNLL-2003/` in IOB2 tagging sceheme (look [here](https://github.com/Franck-Dernoncourt/NeuroNER/tree/4cbfc3a1b4c4a5242e1cfbaea48d6f7e972e8881/data/conll2003/en) for instance ;) and [here](https://github.com/kermitt2/delft/tree/master/delft/utilities). The CONLL 2003 dataset (English) is the default dataset and English is the default language, but you can also indicate it explicitly as parameter with `--dataset-type conll2003` and specifying explicitly the language `--lang en`.

For training and evaluating following the traditional approach (training with the train set without validation set, and evaluating on test set), use:

```sh
> python3 delft/applications/nerTagger.py --dataset-type conll2003 train_eval
```

Some recent works like (Chiu & Nichols, 2016) and (Peters and al., 2017) also train with the validation set, leading obviously to a better accuracy (still they compare their scores with scores previously reported trained differently, which is arguably a bit unfair - this aspect is mentioned in (Ma & Hovy, 2016)). To train with both train and validation sets, use the parameter `--train-with-validation-set`:

> python3 delft/applications/nerTagger.py --dataset-type conll2003 --train-with-validation-set train_eval

Note that, by default, the BidLSTM-CRF model is used. (Documentation on selecting other models and setting hyperparameters to be included here !)

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

Using BERT architecture for sequence labelling (pre-trained transformer with fine-tuning), for instance here the `bert-base-en`, cased, pre-trained model, use:

```sh
> python3 delft/applications/nerTagger.py --architecture BERF_CRF --dataset-type conll2003 --fold-count 10 --transformer bert-base-en train_eval
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
> python3 delft/applications/nerTagger.py --dataset-type conll2003 --file-in data/test/test.ner.en.txt --architecture BERT_CRF_FEATURES --transformer bert-base-en tag
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


The performance is as follow, for the BiLSTM-CRF architecture and fasttext `wiki.fr` embeddings, with a f-score of __91.01__ averaged over 10 training:

```text
average over 10 folds
  macro f1 = 0.9100881012386587
  macro precision = 0.9048633201198737
  macro recall = 0.9153907496012759 

** Worst ** model scores - 

                  precision    recall  f1-score   support

      <location>     0.9467    0.9647    0.9556       368
   <institution>     0.8621    0.8333    0.8475        30
      <artifact>     1.0000    0.5000    0.6667         4
  <organisation>     0.9146    0.8089    0.8585       225
        <person>     0.9264    0.9522    0.9391       251
      <business>     0.8463    0.8936    0.8693       376

all (micro avg.)     0.9040    0.9083    0.9061      1254

** Best ** model scores - 

                  precision    recall  f1-score   support

      <location>     0.9439    0.9592    0.9515       368
   <institution>     0.8667    0.8667    0.8667        30
      <artifact>     1.0000    0.5000    0.6667         4
  <organisation>     0.8813    0.8578    0.8694       225
        <person>     0.9453    0.9641    0.9546       251
      <business>     0.8706    0.9122    0.8909       376

all (micro avg.)     0.9090    0.9242    0.9166      1254
```

With frELMo:

```text
average over 10 folds
    macro f1 = 0.9209397554337976
    macro precision = 0.91949107960079
    macro recall = 0.9224082934609251 

** Worst ** model scores - 

                  precision    recall  f1-score   support

  <organisation>     0.8704    0.8356    0.8526       225
        <person>     0.9344    0.9641    0.9490       251
      <artifact>     1.0000    0.5000    0.6667         4
      <location>     0.9173    0.9647    0.9404       368
   <institution>     0.8889    0.8000    0.8421        30
      <business>     0.9130    0.8936    0.9032       376

all (micro avg.)     0.9110    0.9147    0.9129      1254

** Best ** model scores - 

                  precision    recall  f1-score   support

  <organisation>     0.9061    0.8578    0.8813       225
        <person>     0.9416    0.9641    0.9528       251
      <artifact>     1.0000    0.5000    0.6667         4
      <location>     0.9570    0.9674    0.9622       368
   <institution>     0.8889    0.8000    0.8421        30
      <business>     0.9016    0.9255    0.9134       376

all (micro avg.)     0.9268    0.9290    0.9279      1254
```

For historical reason, we can also consider a particular split of the FTB corpus into train, dev and set set and with a forced tokenization (like the old CoNLL 2013 NER), that was used in previous work for comparison. Obviously the evaluation is dependent to this particular set and the n-fold cross validation is a much better practice and should be prefered (as well as a format that do not force a tokenization). For using the forced split FTB (using the files `ftb6_dev.conll`, `ftb6_test.conll` and `ftb6_train.conll` located under `delft/data/sequenceLabelling/leMonde/`), use as parameter `--dataset-type ftb_force_split`:

```sh
> python3 delft/applications/nerTagger.py --lang fr --dataset-type ftb_force_split --fold-count 10 train_eval
```

which gives for the BiLSTM-CRF architecture and fasttext `wiki.fr` embeddings, a f-score of __86.37__ averaged over 10 training:

```
average over 10 folds
                    precision    recall  f1-score   support

      Organization     0.8410    0.7431    0.7888       311
            Person     0.9086    0.9327    0.9204       205
          Location     0.9219    0.9144    0.9181       347
           Company     0.8140    0.8603    0.8364       290
  FictionCharacter     0.0000    0.0000    0.0000         2
           Product     1.0000    1.0000    1.0000         3
               POI     0.0000    0.0000    0.0000         0
           company     0.0000    0.0000    0.0000         0

  macro f1 = 0.8637
  macro precision = 0.8708
  macro recall = 0.8567 


** Worst ** model scores -
                  precision    recall  f1-score   support

    Organization     0.8132    0.7138    0.7603       311
        Location     0.9152    0.9020    0.9086       347
         Company     0.7926    0.8172    0.8048       290
          Person     0.9095    0.9317    0.9205       205
         Product     1.0000    1.0000    1.0000         3
FictionCharacter     0.0000    0.0000    0.0000         2

all (micro avg.)     0.8571    0.8342    0.8455      1158


** Best ** model scores -
                  precision    recall  f1-score   support

    Organization     0.8542    0.7910    0.8214       311
        Location     0.9226    0.9280    0.9253       347
         Company     0.8212    0.8552    0.8378       290
          Person     0.9095    0.9317    0.9205       205
         Product     1.0000    1.0000    1.0000         3
FictionCharacter     0.0000    0.0000    0.0000         2

all (micro avg.)     0.8767    0.8722    0.8745      1158
```

With frELMo:

```
average over 10 folds
                    precision    recall  f1-score   support

      Organization     0.8605    0.7752    0.8155       311
            Person     0.9227    0.9371    0.9298       205
          Location     0.9281    0.9432    0.9356       347
           Company     0.8401    0.8779    0.8585       290
  FictionCharacter     0.1000    0.0500    0.0667         2
           Product     0.8750    1.0000    0.9286         3
               POI     0.0000    0.0000    0.0000         0
           company     0.0000    0.0000    0.0000         0

  macro f1 = 0.8831
  macro precision = 0.8870
  macro recall = 0.8793 


** Worst ** model scores -
                  precision    recall  f1-score   support

        Location     0.9366    0.9366    0.9366       347
    Organization     0.8309    0.7428    0.7844       311
          Person     0.9268    0.9268    0.9268       205
         Company     0.8179    0.8828    0.8491       290
         Product     0.7500    1.0000    0.8571         3
FictionCharacter     0.0000    0.0000    0.0000         2

all (micro avg.)     0.8762    0.8679    0.8720      1158


** Best ** model scores -
                  precision    recall  f1-score   support

        Location     0.9220    0.9539    0.9377       347
    Organization     0.8777    0.7846    0.8285       311
          Person     0.9187    0.9366    0.9275       205
         Company     0.8444    0.9172    0.8793       290
         Product     1.0000    1.0000    1.0000         3
FictionCharacter     0.0000    0.0000    0.0000         2

all (micro avg.)     0.8900    0.8946    0.8923      1158
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
