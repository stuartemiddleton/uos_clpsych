## University of Southampton CLPsych 2022 software submission

This work can be cited as:
Tayyaba Azim, Loitongbam Gyanendro Singh, Stuart E. Middleton,
__*Detecting Moments of Change and Suicidal Risks in Longitudinal User Texts Using Multi-task Learning*__,
CLPsych-2022 @ NAACL 2022

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

