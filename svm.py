import json
import os

import numpy as np
from nltk.corpus import reuters as rt
from sklearn.svm import SVC
from utils import OUTPUTS_DIR, TESTING_SET, TRAINING_SET

svmmodel = SVC(kernel='linear')

# remove things which have multiple classes
TRAINING_SET = list(filter(lambda x: len(rt.categories(x)) == 1, TRAINING_SET))
TESTING_SET = list(filter(lambda x: len(rt.categories(x)) == 1, TESTING_SET))

X = []
for i in TRAINING_SET:
    with open(os.path.join(OUTPUTS_DIR, i)) as f:
        X.append(json.load(f))
X = np.array(X)

y = []  # Yes, this is a small letter y. No, that is not a mistake.
for i in TRAINING_SET:
    y.append(rt.categories(i))
y = np.array(y)

svmmodel.fit(X, y.ravel())

Z = []
for i in TESTING_SET:
    with open(os.path.join(OUTPUTS_DIR, i)) as f:
        Z.append(json.load(f))
Z = np.array(Z)

total = list(map(lambda x: x[0] == x[1], list(zip(svmmodel.predict(Z), map(lambda x: x[0], map(lambda x: rt.categories(x), TESTING_SET))))))

print(sum(total) / len(total))
