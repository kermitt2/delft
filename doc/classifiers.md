# Snippet classification with DeLFT

We used DeLFT for creating classifiers for various snippet types: citation contexts, software mention contexts, dataset sentences, etc. 
Here are some application examples for this usage. 

 #### Toxic comment classification

The dataset of the [Kaggle Toxic Comment Classification challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) can be found here: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data

This is a multi-label regression problem, where a Wikipedia comment (or any similar short texts) should be associated to 6 possible types of toxicity (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).

To launch the training:

```sh
> python3 delft/applications/toxicCommentClassifier.py train
```

For training with n-folds, use the parameter `--fold-count`:

```sh
> python3 delft/applications/toxicCommentClassifier.py train --fold-count 10
```

After training (1 or n-folds), to process the Kaggle test set, use:

```sh
> python3 delft/applications/toxicCommentClassifier.py test
```

To classify a set of comments:

```sh
> python3 delft/applications/toxicCommentClassifier.py classify
```

#### Citation classification

We use the dataset developed and presented by A. Athar in the following article:

[7] Awais Athar. "Sentiment Analysis of Citations using Sentence Structure-Based Features". Proceedings of the ACL 2011 Student Session, 81-87, 2011. http://www.aclweb.org/anthology/P11-3015

For a given scientific article, the task is to estimate if the occurrence of a bibliographical citation is positive, neutral or negative given its citation context. Note that the dataset, similarly to the Toxic Comment classification, is highly unbalanced (86% of the citations are neutral).

In this example, we formulate the problem as a 3 class regression (`negative`. `neutral`, `positive`). To train the model:

```sh
> python3 delft/applications/citationClassifier.py train
```

with n-folds:

```sh
> python3 delft/applications/citationClassifier.py train --fold-count 10
```

Training and evalation (ratio) with 10-folds:

```sh
> python3 delft/applications/citationClassifier.py train_eval --fold-count 10
```

which should produce the following evaluation (using the 2-layers Bidirectional GRU model `gru`):

```
Evaluation on 896 instances:
                   precision        recall       f-score       support
      negative        0.1494        0.4483        0.2241            29
       neutral        0.9653        0.8058        0.8784           793
      positive        0.3333        0.6622        0.4434            74
```

Similarly as other scripts, use `--architecture` to specify an alternative DL architecture, for instance SciBERT:

```sh
> python3 delft/applications/citationClassifier.py train_eval --architecture scibert
```

```
Evaluation on 896 instances:
                   precision        recall       f-score       support
      negative        0.1712        0.6552        0.2714            29
       neutral        0.9740        0.8020        0.8797           793
      positive        0.4015        0.7162        0.5146            74
```

To classify a set of citation contexts with default model (2-layers Bidirectional GRU model `gru`):

```sh
> python3 delft/applications/citationClassifier.py classify
```

which will produce some JSON output like this:

```json
{
    "model": "citations",
    "date": "2018-05-13T16:06:12.995944",
    "software": "DeLFT",
    "classifications": [
        {
            "negative": 0.001178970211185515,
            "text": "One successful strategy [15] computes the set-similarity involving (multi-word) keyphrases about the mentions and the entities, collected from the KG.",
            "neutral": 0.187219500541687,
            "positive": 0.8640883564949036
        },
        {
            "negative": 0.4590276777744293,
            "text": "Unfortunately, fewer than half of the OCs in the DAML02 OC catalog (Dias et al. 2002) are suitable for use with the isochrone-fitting method because of the lack of a prominent main sequence, in addition to an absence of radial velocity and proper-motion data.",
            "neutral": 0.3570767939090729,
            "positive": 0.18021513521671295
        },
        {
            "negative": 0.0726129561662674,
            "text": "However, we found that the pairwise approach LambdaMART [41] achieved the best performance on our datasets among most learning to rank algorithms.",
            "neutral": 0.12469841539859772,
            "positive": 0.8224021196365356
        }
    ],
    "runtime": 1.202
}

```
