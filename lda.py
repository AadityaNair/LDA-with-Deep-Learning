import logging
import os

import gensim
from nltk.corpus import reuters as rt
from utils import TRAINING_SET, preprocess_document

NUM_TOPICS = int(os.environ.get('NUM_TOPICS', 20))
WORKERS = int(os.environ.get('WORKERS', 3))

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

lda = gensim.models.ldamulticore.LdaMulticore


words_list = list(
    map(
        preprocess_document,
        map(
            lambda x: rt.raw(x),
            TRAINING_SET
        )
    )
)

dictionary = gensim.corpora.Dictionary(words_list)
bow_list = list(map(lambda x: dictionary.doc2bow(x), words_list))

ldamodel = lda(bow_list, num_topics=NUM_TOPICS, id2word=dictionary, passes=100, workers=WORKERS)
ldamodel.save(os.environ.get('LDA_MODEL', './models/lda/trained_lda.txt'))
