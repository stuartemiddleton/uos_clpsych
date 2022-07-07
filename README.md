# University of Southampton CLPsych 2022 Shared Task Submission
## Detecting Moments of Change and Suicidal Risks in Longitudinal User Texts Using Multi-task Learning
This work describes the classification system proposed for the Computational Linguistics and Clinical Psychology (CLPsych) Shared Task 2022. We propose the use of multitask learning approach with a bidirectional long-short term memory (Bi-LSTM) model for predicting changes in user's mood (Task A) and their suicidal risk level (Task B). The two classification tasks have been solved independently or in an augmented way previously, where the output of one task is leveraged for learning another task, however this work proposes an 'all-in-one' framework that jointly learns the related mental health tasks. Our experimental results (ranked top for task A) suggest that the proposed multi-task framework outperforms the alternative single-task frameworks submitted to the challenge and evaluated via the timeline based and coverage based performance metrics shared by the organisers. We also assess the potential of using various types of feature embedding schemes that could prove useful in initialising the Bi-LSTM model for better multitask learning in the mental health domain.

This work is part of the UKRI TAS Hub SafeSpacesNLP project https://www.tas.ac.uk/safespacesnlp/ and supported by the Engineering and Physical Sciences Research Council (EP/V00784X/1). If you use any of the resources in this repository, please cite it as:

Tayyaba Azim, Loitongbam Gyanendro Singh, Stuart E. Middleton,
__*Detecting Moments of Change and Suicidal Risks in Longitudinal User Texts Using Multi-task Learning*__,
CLPsych-2022 @ NAACL, July 10–15, 2022.
```
@inproceedings{DBLP:conf/acl-clpsych/Azim2022,
  author    = {Tayyaba Azim and
               Loitongbam Gyanendro Singh and 
               Stuart E. Middleton},
  editor    = { },
  title     = {Detecting Moments of Change and Suicidal Risks in Longitudinal User Texts Using Multi-task Learning},
  booktitle = {Proceedings of the Eight Workshop on Computational Linguistics and
               Clinical Psychology: : Mental Health in the Face of Change, CLPsych@NAACL
               Hybrid: Online+ Seattle, Washington, July 10–15, 2022 },
  pages     = {xx-xx},
  publisher = {Association for Computational Linguistics},
  year      = {2022},
  url       = {https://doi.org/10.xxxxxxx/v1/wxy-xxxx},
  doi       = {10.1xxxxxxxxxx/v1/w18-xxxxxxxxx},
  timestamp = {Fri, 06 Aug 2021 01:00:00 +0200},
  biburl    = {https://dblp.org/rec/conf/acl-clpsych/Azim2022.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
  
  ```

## Proposed Framework
<img src="https://github.com/stuartemiddleton/uos_clpsych/blob/main/image/Pipeline.png" alt="Framework">
<br>

## License

### Data Set: 
The CLPsych data set is proprietary and not shared here. Please contact the competition organisers at clpsych2022-organizers@googlegroups.com to get a copy of its distribution.
### Software: 
 - © Copyright University of Southampton, 2022, Highfield, University Road, Southampton SO17 1BJ.
 - Created By : Tayyaba Azim and Gyanendro Loitongbam
 - Created Date : 2022/05/26
 - Project : SafeSpacesNLP (https://www.tas.ac.uk/safespacesnlp/)

## Installation Requirements Under Ubuntu 20.04LTS 
+ The experiments were run on Dell Precision 5820 Tower Workstation with Nvidia Quadro RTX 6000 24 GB GPU using Nvidia CUDA Toolkit 11.5.
+ Install the following pre-requisite libraries:
```
pip install -U sentence-transformers
pip install gensim
pip install transformers
pip install tensorflow
pip install keras

# tested Ubunti 20.04 LTS, cuda-11.7 and the below python libraries

Package                       Version
----------------------------- --------------------
gensim                        4.0.1
keras                         2.9.0
tensorflow                    2.9.1
transformers                  4.20.1

```
## Pretrained Models Required
+ download [fastText embedding vectors](https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip)
```
cd <uos_clpsych_dir>
mkdir dataset
cd <uos_clpsych_dir>/dataset
wget -O wiki-news-300d-1M.vec.zip https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
unzip wiki-news-300d-1M.vec.zip
```

## Preparing Data Set
The CLPsych 2022 dataset consists of two types of tab delimited CSV files: *timeline_ID*.tsv and *train_users*.json. 

Each *timeline_ID*.tsv contains information about user posts (postid, date, label, title, content) in a particular timeline and 
the filename (*timeline_ID*) is the timeline ID. 

Each *train_users*.json consists of { user_ID : { risk_level : Severe|Moderate|Low, timelines : [timeline_ID, ...] }. 

We merged all the files and prepare a single comma delimited CSV file containing both the informations in *timeline_ID*.tsv and *train_users*.json. The following relational table structure shows how we prepared for the single CSV file.

```
| Timeline_ID | User_ID | User_risk | Content | Post_ID | Post_label |
```

- *Timeline_ID*: This is the timeline ID of the sequence of posts by a user
- *User_ID*: The user id of the above timeline.
- *User_Risk*: It is the user risk label shared with all the timelines of the user above (Severe|Moderate|Low).
- *Content*: This is the total text merged with title and content for each user post.
- *Post_ID*: It is the post id.
- *Post_label*: It is the label of the post to indicate the moment of change (0|IS|ES). IS = switch. ES = escalation.

Save the training and testing sets as *training_dataset.csv* and *testing_dataset.csv* respectively.

## Sentence Embedding Methods
There are two types of sentence embedding methods considered for this study (Please refer to the paper for detail explaination):
+ *sent_emb*: fastText + SBERT 
+ *sent_score_emb*: fastText + SBERT + Task-specific scores

##  Training Models
```
cd <uos_clpsych_dir>
mkdir results
mkdir model

# train Multitask (use nohup to run in background as training can take a few hours)
nohup python codes/CLPsych-multitask_text.py --attention_layer 0 --load_classes dataset/training_classes_index.pkl --training_dataset dataset/training_dataset.csv --testing_dataset dataset/testing_dataset.csv --result_dir results/ --save_model 1 > train_stdout.txt &

# train Multitask-attn
nohup python codes/CLPsych-multitask_text.py --attention_layer 1 --load_classes dataset/training_classes_index.pkl --training_dataset dataset/training_dataset.csv --testing_dataset dataset/testing_dataset.csv --result_dir results/ --save_model 1 > train_stdout.txt &

# train Multitask-score
nohup python codes/CLPsych-multitask_text-score.py --attention_layer 0 --load_classes dataset/training_classes_index.pkl --training_dataset dataset/training_dataset.csv --testing_dataset dataset/testing_dataset.csv --result_dir results/ --save_model 1 > train_stdout.txt &

# train Multitask-attn-score
nohup python codes/CLPsych-multitask_text-score.py --attention_layer 1 --load_classes dataset/training_classes_index.pkl --training_dataset dataset/training_dataset.csv --testing_dataset dataset/testing_dataset.csv --result_dir results/ --save_model 1 > train_stdout.txt &
```
+ *attention_layer*: Flag to define whether the model to be trained is with (1) or without (0) attention layer.
+ *load_classes*: Location to load or save the pickle of training class indices.
+ *training_dataset*: Location of the training dataset file.
+ *testing_dataset*: Location of the testing dataset file.
+ *result_dir*: Location to save the model predicted results.
+ *save_model*: Flag to save model yes (1) or not (0).

## Testing Models
```
# test Multitask
python codes/CLPsych-multitask_text_testing.py --attention_layer 0 --load_classes dataset/training_classes_index.pkl --testing_dataset dataset/testing_dataset.csv --result_dir results/

# test Multitask-attn
python codes/CLPsych-multitask_text_testing.py --attention_layer 1 --load_classes dataset/training_classes_index.pkl --testing_dataset dataset/testing_dataset.csv --result_dir results/

# test Multitask-score
python codes/CLPsych-multitask_text_testing_score.py --attention_layer 0 --load_classes dataset/training_classes_index.pkl --testing_dataset dataset/testing_dataset.csv --result_dir results/

# test Multitask-attn-score
python codes/CLPsych-multitask_text_testing_score.py --attention_layer 1 --load_classes dataset/training_classes_index.pkl --testing_dataset dataset/testing_dataset.csv --result_dir results/
```
+ *Multitask*: model using *sent_emb* 
+ *Multitask-score*: model using *sent_score_emb* 
+ *Multitask-attn*: model with attention layer using *sent_emb*
+ *Multitask-attn-score*: model with attention layer using *sent_score_emb*. 

## Evaluating models on Shared Task 2022 Validation Set Results (run by our team locally)
```
# task a
py codes/evaluate.py results/training_dataset_multitask-taska-with_attention.csv no
py codes/evaluate.py results/training_dataset_multitask-taska-without_attention.csv no
py codes/evaluate.py results/training_dataset_multitask-score-taska-without_attention.csv no
py codes/evaluate.py results/training_dataset_multitask-score-taska-with_attention.csv no

# task b
py codes/evaluate.py results/training_dataset_multitask-taskb-with_attention.csv yes
py codes/evaluate.py results/training_dataset_multitask-taskb-without_attention.csv yes
py codes/evaluate.py results/training_dataset_multitask-score-taskb-without_attention.csv yes
py codes/evaluate.py results/training_dataset_multitask-score-taskb-with_attention.csv yes

```
**Task A: Moments of Change**
| Model | Precision | Recall | F1 |
| ----- | --------- | ------ | -- |
| *Multitask*	| 0.582	| 0.717	| 0.629	| 
| *Multitask-attn*	| 0.663	| 0.697	| 0.676	| 
| *Multitask-score*	| 0.680	| 0.760	| 0.713	| 
| *Multitask-attn-score*	| 0.674	| 0.800	| 0.724	| 

**Task B: Suicidal Risk Levels**
| Model | Precision | Recall | F1 |
| ----- | --------- | ------ | -- |
| *Multitask*	| 0.352	| 0.327	| 0.335	| 
| *Multitask-attn*	|  0.408	| 0.378	| 0.388	| 
| *Multitask-score*	|  0.355	| 0.331	| 0.334	| 
| *Multitask-attn-score*	| 0.415	| 0.397	| 0.382	| 

## Evaluating models on Shared Task 2022 Test Set Results (run by CLPsych-2022 organisers on our behalf using hidden test labels)
**Post-level metrics (Task-A)**   (table)
| Model | Precision | Recall | F1 |
| ----- | --------- | ------ | -- |
| *Multitask*	| 0.680	| 0.579	| 0.649	| 
| *Multitask-score*	| 0.677	| 0.595	| 0.625	| 
| *Multitask-attn-score*	| 0.680	| 0.579	| 0.607	| 


**User-level metrics (Task-B)**
| Model | Precision | Recall | F1 |
| ----- | --------- | ------ | -- |
| *Multitask-attn-score*	| 0.618	| 0.427	| 0.451	| 


