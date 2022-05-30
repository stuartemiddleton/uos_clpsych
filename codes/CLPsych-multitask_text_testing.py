######################################################################
#
# (c) Copyright University of Southampton, 2022
#
# Copyright in this software belongs to University of Southampton,
# Highfield, University Road, Southampton SO17 1BJ
#
# Created By : Tayyaba Azim, Gyanendro Loitongbam
# Created Date : 2022/05/26
# Project : SafeSpacesNLP
#
######################################################################

import pandas as pd
import numpy as np
import gensim
import nltk
from nltk.corpus import stopwords
import sys
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Activation, Flatten
from keras.layers import Attention, MultiHeadAttention
from keras.models import model_from_json
import pickle
from sentence_transformers import SentenceTransformer

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the models')
    parser.add_argument('--model', help='Model type- 0: without_attention, 1: with_attention')
    parser.add_argument('--load_classes', help='Saved class index file location')
    parser.add_argument('--testing_dataset', help='Testing dataset file location')
    parser.add_argument('--result_dir', help='Store result directory')

    args = parser.parse_args()

    return args


args = parse_args()


selection = args.model
loaded_classes = args.load_classes          #'dataset/CLPsych_dataset/teamdata/training_classes.pkl'
testing_dataset = args.testing_dataset      #"dataset/CLPsych_dataset/teamdata/testing_dataset.csv"
result_save_loc =  args.result_dir          #'dataset/CLPsych_dataset/teamdata/'


if selection == '0':
    model_type = 'without_attention'
elif selection == '1':
    model_type = 'with_attention'

print('You have selected',model_type,'for this experiment')

sv_model = SentenceTransformer('sentence-transformers/nli-roberta-large')
wv_model = gensim.models.KeyedVectors.load_word2vec_format('dataset/wiki-news-300d-1M.vec', binary=False)
stops = set(stopwords.words('english'))

def remove_stopwords(sentence):
    words = sentence.split()
    wordsFiltered = []
    for w in words:
        if w not in stops:
            wordsFiltered.append(w)
    return wordsFiltered

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def get_sentence_emb(text):
    wdim = (wv_model['word'].shape)[0]
    text = preprocess(text)
    words = remove_stopwords(text)
    sentence_emb = np.zeros(wdim)
    for word in words:
        if word in wv_model:
            sentence_emb += wv_model[word]
    sentence_emb = sentence_emb/len(words)
    return sentence_emb

#Hyperparameter setting
max_len = 122

class PostGetter(object):
    def __init__(self, data):
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t, c, ss, u) for w, p, t, c, ss,u in zip(s["postid"].values.tolist(),
                                                           s["User_Risk"].values.tolist(),
                                                           s["label"].values.tolist(),
                                                           s["Content"].values.tolist(),
                                                           s["Session_ID"].values.tolist(),
                                                           s["User_ID"].values.tolist())]
        self.grouped = self.data.groupby("Session_ID").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self,sessionID):
        try:
            s = self.grouped["Session: {}".format(sessionID)]
            return s
        except:
            return None

f = open(loaded_classes, 'rb')
(tag2idx,risk2idx) = pickle.load(f)
f.close()

idx2tag = {tag2idx[t]: t for t in tag2idx}
idx2risk = {risk2idx[r]: r for r in risk2idx}


# load json and create model
json_file = open("model/model_multitask-sentence-transformer-130-"+model_type+".json", 'r')
loaded_model_json = json_file.read()
json_file.close()

model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model/model_multitask-sentence-transformer-130-"+model_type+".h5")


'''Testing phases'''

embeddings = sv_model.encode(['This is a test'], device='cpu')
wdim = (wv_model['word'].shape)[0]
wdim = wdim+embeddings.shape[1]
sentence_emb = np.zeros(wdim)+9



data = pd.read_csv(testing_dataset)
data = data.fillna(method="ffill")
sessions = set(list(data['Session_ID']))
sentences = data[['postid','Content']]
sentences_id = sentences.set_index('postid')['Content'].to_dict()

getter = PostGetter(data)
sentences = getter.sentences

postids = list(set(data["postid"].values))
postids.append("ENDPAD")
n_postids = len(postids)

post2idx = {w: i for i, w in enumerate(postids)}
idx2post = {i: w for i, w in enumerate(postids)}

X = [[post2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_postids - 1)

XXt = []
for sess in X:
    ses_pos = []
    for pid in sess:
        pidd = idx2post[pid]
        if pidd != 'ENDPAD':
            sent = get_sentence_emb(sentences_id[pidd])
            embeddings = sv_model.encode([sentences_id[pidd]], device='cpu')
            sent = np.concatenate((sent, embeddings[0]), axis=None)
            ses_pos.append(sent)
        else:
            ses_pos.append(sentence_emb)
    ses_pos = np.asarray(ses_pos)
    XXt.append(ses_pos)

XXt = np.asarray(XXt)

result = model.predict(XXt)

user = []
pred_label = []
post_id = []
time_line = []
content = []

user_b = []
pred_b = []

for si,sess in enumerate(sentences):
    us_id = sess[0][5]
    lab_b = sess[0][1]
    pd_b = idx2risk[np.argmax(result[1][si])]
    
    user_b.append(us_id)
    pred_b.append(pd_b)

    for pi, post in enumerate(sess): 
         post_id.append(post[0])
         content.append(post[3])
         time_line.append((post[4].split())[1])
         user.append(post[5])
         pred_a = idx2tag[np.argmax(result[0][si][pi])]
         pred_label.append(pred_a)

taska_pd = pd.DataFrame({'Timeline': time_line, 'Postid': post_id, 'Content': content,'Pred_Label': pred_label})
taska_pd.to_csv(result_save_loc+'testing_dataset_multitask-taska-'+model_type+'.csv', index=False)

taskb_pd = pd.DataFrame({'UserID': user_b, 'Pred_Label': pred_b})

multi_user_func = lambda s: [(u,p) for u, p in zip(s["UserID"].values.tolist(),
                                                   s["Pred_Label"].values.tolist())]
grouped = taskb_pd.groupby("UserID").apply(multi_user_func)


user_bi = []
pred_bi = []
for i in grouped:
    user_bi.append(i[0][0])
    classes = {}
    for tm in i:
        if tm[1] not in classes:
            classes[tm[1]] = 1
        classes[tm[1]] += 1

    classes = {k: v for k, v in sorted(classes.items(), reverse=True, key=lambda item: item[1])}

    if 'Severe' in classes:
        f_risk = 'Severe'
    elif 'Moderate' in classes:
        f_risk = 'Moderate'
    else:
        f_risk = 'Low'

    pred_bi.append(f_risk)

taskb_pd_grouped = pd.DataFrame({'userID': user_bi, 'label': pred_bi})
taskb_pd_grouped.to_csv(result_save_loc+'testing_dataset_multitask-taskb-'+model_type+'.csv', index=False)



    















