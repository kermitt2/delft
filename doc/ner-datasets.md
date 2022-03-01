# NER Datasets in DeLFT

## CoNLL-2003 and Ontonotes 5.0 for CoNLL-2012 datasets 

This page provides some details and precisions about the CoNLL training data used for training the NER models. Most information here are well-known, and we compile them for reference. We provide also two simple scripts to get these standard datasets in [IOB2](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) label scheme (the most common one) used by DeLFT. 

### CoNLL 2003

* You need first the [Reuters Corpus](https://trec.nist.gov/data/reuters/reuters.html) which is free of charge for research purposes. 

* Second get the annotations and assembling scripts [here](https://www.clips.uantwerpen.be/conll2003/ner/)

* Follow the instructions for assembling the CoNLL 2003 annotated corpus, you will get respectively the train, validation and test files: `eng.train`, `eng.testa`, `eng.testb`

* The generated files contain NER labels together with syntactic labels. In addition the NER labels follow the IOB scheme (`B-` prefix appears only when two distinct named entities of the same class occur successively). For generating the NER annotation in IOB2 format, use:

> python3 utilities/Utilities.py --dataset-type conll2003 --data-path /home/lopez/resources/CoNLL-2003/eng.train --output-path /home/lopez/resources/CoNLL-2003/iob2/eng.train 

`--data-path` is the path to one of the file generated with the standard assembling scripts

`--output-path` is the path of the file converted into IOB2 format to be written 

The resulting IOB2 files are very similar to these [one](https://github.com/Franck-Dernoncourt/NeuroNER/tree/4cbfc3a1b4c4a5242e1cfbaea48d6f7e972e8881/data/conll2003/en) for example (but the NeuroNER version removes some document information).

### CoNLL 2012

* You need first the Ontonotes 5.0 corpus available at the [LDC](https://catalog.ldc.upenn.edu/LDC2013T19), free of charge for research purposes. 

* Then get the annotation and assembling scripts [here](http://cemantix.org/data/ontonotes.html) updated for Ontonotes 5.0.

* The assembling script will generate a hierarchy of completely annotated files with gold-standard quality. 

* Similarly as for CoNLL-2003 dataset, you can generate the annotated data with only NER labels in IOB2 scheme with the following command: 

> python3 utilities/Utilities.py --dataset-type conll2012 --data-path /home/lopez/resources/ontonotes/conll-2012/ --output-path /home/lopez/resources/ontonotes/conll-2012/iob2/

`--data-path` is the path to the root of the CoNLL-2012 hierarchy of assembled files

`--output-path` is the path where the files converted into IOB2 format will be written 

Three files will be written `eng.train`, `eng.dev`, `eng.test` corresponding exactly to the list of documents of the official CoNLL-2012 for the train, development and test sets (a large subset of the whole Ontonotes corpus). Similarly as all the previous evaluations we have seen so far and for consistency with the previously reported scores, we exclude the PT set (English translation of the New Testament, ≈200k), see for instance (Durrett and Klein, 2014) and (Chiu and Nichols, 2016).

__(Durrett and Klein, 2014)__ Greg Durrett, Dan Klein. "A joint model for entity analysis: Coreference, typing, and linking", 2014. http://www.aclweb.org/anthology/Q14-1037

__(Chiu and Nichols, 2016)__ Jason P. C. Chiu, Eric Nichols. "Named Entity Recognition with Bidirectional LSTM-CNNs". 2016. https://arxiv.org/abs/1511.08308
