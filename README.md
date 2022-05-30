# University of Southampton CLPsych 2022 software submission

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

## Installation under Ubuntu 20.04LTS 
+ *The experiment was done on Dell Precision 5820 Tower Workstation with Nvidia Quadro RTX 6000 24 GB GPU using Nvidia CUDA Toolkit 11.5*
+ *Install the following pre-requisite libraries*
```
pip install sentence_transformers
pip install gensim
pip install transformer
pip install tensorflow
pip install keras
```



## Pretrained models
+ [fastText embedding vectors](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip)

## Preparing dataset
The dataset provided by the CLPsych organiser cannot be shared. To get the dataset, you can communicate with the [CLPsych organiser](https://clpsych.org/). The dataset consist of two types of files: *file-type-A*.csv and *file-type-B*.json. Each *file-type-A*.csv contains information about user posts (with time, user-id, posts labels) in a particular timeline and the filename (*file-type-A*) is the timeline ID. Each *file-type-B*.json consist of user-id, timelines, and user-risk label and the filename (*file-type-B*) is the user ID. We merge all the files and prepare a single CSV file containing both the informations in *file-type-A* and *file-type-B*. The following relational table structure shows how we prepared for the single CSV file merging all the information of *file-type-A* and *file-type-B*.

| Timeline_ID | User_ID | User_risk | Content | Post_ID | Post_label |

- Timeline_ID: This is the timeline ID of the sequence of posts by a user
- User_ID: The user id of the above timeline.
- User_Risk: It is the user risk label shared with all the timelines of the user above.
- Content: This is the total text merged with title and content for each user post.
- Post_ID: It is the post id.
- Post_label: It is the label of the post to indicate the moment of change.

Save the training and testing sets as *training_dataset.csv* and *testing_dataset.csv* respectively.

## Train models
```
python CLPsych-multitask_text.py --model 0 --load_classes dataset/CLPsych_dataset/teamdata/training_classes.pkl --training_dataset dataset/CLPsych_dataset/teamdata/training_dataset.csv --testing_dataset dataset/CLPsych_dataset/teamdata/testing_dataset.csv --result_dir dataset/CLPsych_dataset/teamdata/ --save_model 0
```
+ *model*: Flag to define whether the model to be trained is with (1) or without (0) attention layer.
+ *load_classes*: Location to load or save the training class indices.
+ *training_dataset*: Location of the training dataset.
+ *testing_dataset*: Location of the testing dataset.
+ *result_dir*: Location to save the model predicted results.
+ *save_model*: Flag to save model yes (1) or not (0).


## Testing models

```
python CLPsych-multitask_text_testing.py --model 0 --load_classes dataset/CLPsych_dataset/teamdata/training_classes.pkl --testing_dataset dataset/CLPsych_dataset/teamdata/testing_dataset.csv --result_dir dataset/CLPsych_dataset/teamdata/
```

## Sentence embedding methods
There are two types of sentence embedding methods considered for this study (Please refer to the paper for detail explaination):
+ *sent_emb*: fastText + SBERT 
+ *sent_score_emb*: fastText + SBERT + Task-specific scores

## Evaluation models
+ *Multitask*: model using *sent_emb* 
+ *Multitask-score*: model using *sent_score_emb* 
+ *Multitask-attn*: model with attention layer using *sent_emb*
+ *Multitask-attn-score*: model with attention layer using *sent_score_emb*. 

## Shared Task 2022 Validation set eval result

Moments of Change 
| Model | Precision | Recall | F1 |
| ----- | --------- | ------ | -- |
| *Multitask*	| 0.582	| 0.717	| 0.629	| 
| *Multitask-attn*	| 0.663	| 0.697	| 0.676	| 
| *Multitask-score*	| 0.680	| 0.760	| 0.713	| 
| *Multitask-attn-score*	| 0.674	| 0.800	| 0.724	| 




Suicidal Risk Levels
| Model | Precision | Recall | F1 |
| ----- | --------- | ------ | -- |
| *Multitask*	| 0.352	| 0.327	| 0.335	| 
| *Multitask-attn*	|  0.408	| 0.378	| 0.388	| 
| *Multitask-score*	|  0.355	| 0.331	| 0.334	| 
| *Multitask-attn-score*	| 0.415	| 0.397	| 0.382	| 


## Shared Task 2022 Test set eval result

Post-level metrics (Task-A)
| Model | Precision | Recall | F1 |
| ----- | --------- | ------ | -- |
| *Multitask*	| 0.680	| 0.579	| 0.649	| 
| *Multitask-score*	| 0.677	| 0.595	| 0.625	| 
| *Multitask-attn-score*	| 0.680	| 0.579	| 0.607	| 


Task-B
| Model | Precision | Recall | F1 |
| ----- | --------- | ------ | -- |
| *Multitask-attn-score*	| 0.618	| 0.427	| 0.451	| 


