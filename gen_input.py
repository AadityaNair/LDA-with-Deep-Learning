import json
import logging
import os

import gensim
from nltk.corpus import reuters as rt
from utils import INPUTS_DIR, TESTING_SET, TRAINING_SET, preprocess_document

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

ldamodel = gensim.models.LdaMulticore.load(os.environ.get('LDA_MODEL', './models/lda/trained_lda.txt'))  # can be any lda. I only want the dictionary
dictionary = ldamodel.id2word

for i in TRAINING_SET + TESTING_SET:
    logging.info(i)
    bow = dictionary.doc2bow(preprocess_document(rt.raw(i)))
    with open(os.path.join(INPUTS_DIR, i), 'w+') as f:
        json.dump(dict(bow), f)
