import pandas as pd
import numpy as np
import re
import spacy
import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn import svm
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, GlobalMaxPool1D, Dropout, Conv1D
from keras.callbacks import EarlyStopping, ModelCheckpoint

def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    '''
    Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
    http://stackoverflow.com/q/32239577/395857
    '''
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/float(len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def print_score(y_pred, clf):
    print("Clf: ", clf.__class__.__name__)
    print("Hamming loss: {}".format(hamming_loss(y_pred, y_val)))
    print("Hamming score: {}".format(hamming_score(y_pred, y_val)))
    print("---")  

def clean_text(text):
    text = re.sub(r'<.*?>','',text)
    text = re.sub(r'[^a-zA-Z]'," ", text)
    text = ' '.join(text.split())
    return text

pd.set_option('display.max_colwidth',200)
data = pd.read_hdf('D:\\NLP\AnalyticsVidhya-NLP\\Handouts_v4\Project - Building Auto-Tagging System\\auto_tagging_data_v2.h5')
data['Text'] = data['Title'] +" "+ data['Body']

data['Text'] = data['Text'].apply(lambda x : clean_text(x))
data['Text'] = data['Text'].str.lower()


#Reshape Target Variable
mb = MultiLabelBinarizer()
mb.fit(data.Tags)
y = mb.transform(data.Tags)

count_vect = CountVectorizer()
x_counts = count_vect.fit_transform(data.Text)
tfidf_transform = TfidfTransformer()
x_tfidf = tfidf_transform.fit_transform(x_counts)
x_train, x_val, y_train, y_val = train_test_split(x_tfidf, y, test_size=0.2, random_state=9)

print("model is splitted")

#LogisticRegression
model = OneVsRestClassifier(LogisticRegression())
model.fit(x_train, y_train)
preds = model.predict(x_val)
print_score(preds, LogisticRegression())

# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(data['Text'])
# vocab_size = len(tokenizer.word_index)+1
# sequences = tokenizer.texts_to_sequences(data['Text'])
# max_length = 125
# padded_seq = pad_sequences(sequences, maxlen=max_length) 

# x_train, x_val, y_train, y_val = train_test_split(padded_seq, y, test_size=0.2, random_state=9)

# model = load_model('model-conv1d_v1.h5')
# def infer_tags(q):
    # q = clean_text(q)
    # q = q.lower()
    # q_seq = tokenizer.texts_to_sequences([q])
    # q_seq_padded = pad_sequences(q_seq, maxlen=300)
    # q_pred = model.predict(q_seq_padded)
    # q_pred = (q_pred >= 0.3).astype(int)
    
    # return multilabel_binarizer.inverse_transform(q_pred)
# # give new question
# new_q = "Regression line in ggplot doesn't match computed regression Im using R and created a chart using ggplot2. I then create a regression so I can make some predicitions I pass my data frame of to the predict function predict(regression, Measures) I'd expect the predictions to be the same as if I used the regression line on the chart, but they aren't the same. Why would this be the case? Is there a setting in ggplot or is my expectation incorrect?"

# # get tags
# print(infer_tags(new_q))