# Save / Load File
import dill
import pickle

# Load Vectors
from gensim.models import KeyedVectors

# Utility
import numpy as np
from utils.use import tag_html_format, prepocess_text

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Keras Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input ,LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D
from tensorflow.keras.layers import concatenate, SpatialDropout1D, GlobalMaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow import keras

model_path = "./trained_model/LSTM/"

thai2fit_model = KeyedVectors.load_word2vec_format('./thai2vec/thai2vecNoSym.bin',binary=True)
thai2fit_weight = thai2fit_model.vectors

thai2dict = {}

for word in thai2fit_model.index_to_key:
    thai2dict[word] = thai2fit_model[word]

all_thai2dict = sorted(set(thai2dict))
thai2dict_to_ix = dict((c, i) for i, c in enumerate(thai2dict)) #convert thai2fit to index 
ix_to_thai2dict = dict((v,k) for k,v in thai2dict_to_ix.items())  #convert index to thai2fit

n_thai2dict = len(thai2dict_to_ix)

with open(model_path+'nerdict.pickle', 'rb') as nerdict:
    ner_to_ix = pickle.load(nerdict)

ix_to_ner = dict((v,k) for k,v in ner_to_ix.items())  #convert index to ner
n_tag = len(ner_to_ix)

with open(model_path+'chardict.pickle', 'rb') as chardict:
    char2idx = pickle.load(chardict)

n_chars = len(char2idx)

max_len = 400
max_len_char = 32

character_LSTM_unit = 32
char_embedding_dim = 32
main_lstm_unit = 256 

def prepare_sequence_word(list_sent):
    idxs = list()
    for word in list_sent:
        if word in thai2dict:
            idxs.append(thai2dict_to_ix[word])
        else:
            idxs.append(thai2dict_to_ix["unknown"]) #Use UNK tag for unknown word
    return idxs

def prepare_sequence_target(input_label):
    idxs = [ner_to_idx[BIO] for BIO in input_label]
    return idxs

#word Input
word_in = Input(shape=(max_len), name='word_input_')

#word Enbedding Using Thai2Fit
word_embeddings = Embedding(input_dim=n_thai2dict, output_dim=400, weights = [thai2fit_weight], input_length=max_len,
                                               mask_zero=False, trainable=False, name="word_embedding")(word_in)

# Character Input
char_in = Input(shape=(max_len, max_len_char,), name='char_input')

# Character Embedding
emb_char = TimeDistributed(Embedding(input_dim=n_chars, output_dim=char_embedding_dim, 
                           input_length=max_len_char, mask_zero=False))(char_in)

# Character Sequence to Vector via BiLSTM
char_enc = TimeDistributed(LSTM(units=character_LSTM_unit, return_sequences=False))(emb_char)

# Concatenate All Embedding
all_word_embeddings = concatenate([word_embeddings, char_enc])
all_word_embeddings = SpatialDropout1D(0.3)(all_word_embeddings)

main_lstm = LSTM(units=main_lstm_unit, return_sequences=True,)(all_word_embeddings)
dens = TimeDistributed(Dense(100, activation="relu"))(main_lstm)
out = Dense(n_tag, activation="softmax")(dens)
model = keras.Model(inputs=[word_in, char_in], outputs=[out])

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.001))

model.load_weights(model_path+"model_LSTM.hdf5")

def convert_word_to_char(predict_word):
    predict_char = []
    sent_seq = []
    for i in range(max_len):
        word_seq = []
        for j in range(max_len_char):    
            try:
                if(predict_word[i][j] in char2idx):
                    word_seq.append(char2idx.get(predict_word[i][j]))
                else:
                    word_seq.append(char2idx.get("unknown"))
            except:
                word_seq.append(char2idx.get("pad"))
        sent_seq.append(word_seq)
    predict_char.append(np.array(sent_seq))
    
    return predict_char

def prediction(text):
    predict_sent = prepocess_text(text)
    len_word = len(predict_sent)
    predict_word = []
    predict_word = [prepare_sequence_word(predict_sent)]
    predict_word = pad_sequences(maxlen=max_len, sequences=predict_word, value=thai2dict_to_ix["pad"], padding='post', truncating='post')

    predict_char = convert_word_to_char(predict_sent)
    result_tag = model.predict([predict_word,np.array(predict_char).reshape((len(predict_char),max_len, max_len_char))])
    p = np.argmax(result_tag, axis=-1)
    pred=[i for i in p[0]]
    revert_pred=[ix_to_ner[i] for i in p[0]]
    
    word=predict_sent
    tag=revert_pred[:len_word]
    return word, tag

def call_model_LSTM(text):
    list_word, predict_tag = prediction(text)
    return tag_html_format(zip(list_word, predict_tag))