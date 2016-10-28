import json
import os

import gensim
import keras
import numpy as np
from utils import INPUTS_DIR, OUTPUTS_DIR, TRAINING_SET

ldamodel = gensim.models.ldamulticore.LdaMulticore.load(os.environ.get('LDA_MODEL', './models/lda/trained_lda.txt'))
dictionary = ldamodel.id2word

X = []
for i in TRAINING_SET:
    with open(os.path.join(INPUTS_DIR, i)) as f:
        d = json.load(f)
    empty = np.zeros(len(dictionary))
    for k, v in d.items():
        empty[int(k)] = float(v)
    X.append(empty)
X = np.array(X)

Y = []
for i in TRAINING_SET:
    with open(os.path.join(OUTPUTS_DIR, i)) as f:
        Y.append(json.load(f))
Y = np.array(Y)

dnnmodel = keras.models.Sequential()
dnnmodel.add(keras.layers.Dense(ldamodel.num_topics * 2, input_dim=len(dictionary), activation='tanh'))
dnnmodel.add(keras.layers.Dense(ldamodel.num_topics, activation='softmax'))
dnnmodel.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

dnnmodel.fit(X, Y, validation_split=0.2, nb_epoch=250, callbacks=[
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=True, mode='auto'),
    keras.callbacks.ModelCheckpoint(os.environ.get('_2NN_MODEL', './models/dnn/trained_2nn.txt'), monitor='val_loss', verbose=True, save_best_only=True, save_weights_only=False, mode='auto')
])
