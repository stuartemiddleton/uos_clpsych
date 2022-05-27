## University of Southampton CLPsych 2022 software submission

This work can be cited as:
Tayyaba Azim, Loitongbam Gyanendro Singh, Stuart E. Middleton,
__*Detecting Moments of Change and Suicidal Risks in Longitudinal User Texts Using Multi-task Learning*__,
CLPsych-2022 @ NAACL 2022

## Abstract
This work describes the classification system proposed for the Computational Linguistics and Clinical Psychology (CLPsych) Shared Task 2022. We propose the use of multitask learning approach with a bidirectional long-short term memory (Bi-LSTM) model for predicting changes in user's mood (Task A) and their suicidal risk level (Task B). The two classification tasks have been solved independently or in an augmented way previously, where the output of one task is leveraged for learning another task, however this work proposes an 'all-in-one' framework that jointly learns the related mental health tasks. Our experimental results (ranked top for task A) suggest that the proposed multi-task framework outperforms the alternative single-task frameworks submitted to the challenge and evaluated via the timeline based and coverage based performance metrics shared by the organisers. We also assess the potential of using various types of feature embedding schemes that could prove useful in initialising the Bi-LSTM model for better multitask learning in the mental health domain.


This work is part of the UKRI TAS Hub SafeSpacesNLP project https://www.tas.ac.uk/safespacesnlp/ and supported by the Engineering and Physical Sciences Research Council (EP/V00784X/1)

<h4>Proposed framework</h4>
<img src="https://github.com/stuartemiddleton/uos_clpsych/blob/main/image/Pipeline.png" alt="Framework">
<br>

# Installation under Ubuntu 20.04LTS

```
TODO install pre-requisite libs
pip install sentence_transformers
pip install gensim

TODO download models
Download pretrained fastText embedding vectors from "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
```

# Train models

```
TODO train models on example data (not real data as we cannot share it)
```

# Infer using models

```
TODO infer moment of change (task A)
TODO infer user at risk (task B)
```

# Latest eval results

TODO results in a table from CLPsych 2022 systems paper

| Model | Precision | Recall | F1 |
| ----- | --------- | ------ | -- |
| Model 1 | 0.97 | 1.0 | 0.98 |
| Model 2 | 1.0 | 1.0 | 1.0 |
| Model 3 | 1.0 | 1.0 | 1.0 |

