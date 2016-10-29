import json
import logging
import os

import gensim
from utils import INPUTS_DIR, OUTPUTS_DIR, TESTING_SET, TRAINING_SET

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

ldamodel = gensim.models.LdaMulticore.load(os.environ.get('LDA_MODEL', './models/lda/trained_lda.txt'))

for i in TRAINING_SET + TESTING_SET:
    with open(os.path.join(INPUTS_DIR, i)) as f:
        bow = json.load(f)
    with open(os.path.join(OUTPUTS_DIR, i), 'w+') as f:
        json.dump(list(map(lambda x: x[1], ldamodel.get_document_topics(list(map(lambda x: (int(x[0]), int(x[1])), bow.items())), minimum_probability=0.0))), f)
