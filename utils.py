import os
import re
from string import punctuation, whitespace

import html

import nltk
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import reuters as rt
from nltk.corpus import stopwords as st
from stop_words import get_stop_words

LOT_OF_STOPWORDS = frozenset(list(STOPWORDS) + get_stop_words('en') + st.words('english'))

TRAINING_SET = list(filter(lambda x: x.startswith('train'), rt.fileids()))
TESTING_SET = list(filter(lambda x: x.startswith('test'), rt.fileids()))

INPUTS_DIR = os.environ.get('INPUTS_DIR', 'inputs')
OUTPUTS_DIR = os.environ.get('OUTPUTS_DIR', 'outputs')

WHITE_PUNC_REGEX = re.compile(r"[%s]+" % re.escape(whitespace + punctuation), re.UNICODE)
lemma = nltk.wordnet.WordNetLemmatizer()


def preprocess_document(document_text):
    """
        1.) Lowercase it all
        2.) Remove HTML Entities
        3.) Split by punctuations to remove them.
        4.) Stem / Lemmaize
        5.) Remove stop words
        6.) Remove unit length words
        7.) Remove numbers
    """
    def is_num(x):
        return not (x.isdigit() or (x[0] == '-' and x[1:].isdigit()))

    return list(
        filter(
            is_num,
            filter(
                lambda x: len(x) > 1,
                filter(
                    lambda x: x not in LOT_OF_STOPWORDS,
                    map(
                        lambda x: lemma.lemmatize(x),
                        re.split(
                            WHITE_PUNC_REGEX,
                            html.unescape(
                                document_text.lower()
                            )
                        )
                    )
                )
            )
        )
    )
