import json
import os

import keras
import gensim
import numpy as np

from utils import INPUTS_DIR, OUTPUTS_DIR, TESTING_SET, TRAINING_SET

ldamodel = gensim .models.ldamulticore.LdaMulticore.load(os.environ.get('LDA_MODEL', './models/dnn/trained_lda.txt'))
dictionary = ldamodel.id2word

dnnmodel = keras.models.load_model(os.environ.get('DNN_MODEL', './models/dnn/trained_dnn.txt'))

X = []
for i in TRAINING_SET + TESTING_SET:
    with open(os.path.join(INPUTS_DIR, i)) as f:
        d = json.load(f)
    empty = np.zeros(len(dictionary))
    for k, v in d.items():
        empty[int(k)] = float(v)
    X.append(empty)
X = np.array(X)

for file, data in zip(TRAINING_SET + TESTING_SET, dnnmodel.predict_on_batch(X)):
    with open(os.path.join(OUTPUTS_DIR, file), 'w+') as f:
        json.dump(data.tolist(), f)
