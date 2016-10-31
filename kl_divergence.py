import keras, gensim
from nltk.corpus import reuters as rt
import numpy as np
from utils import preprocess_document

for ntopics in range(10,110,10):
    print("For number of topics: ", ntopics)

    lda_model = gensim.models.LdaMulticore.load('./models/lda/trained_lda_'+str(ntopics)+'.txt')
    dnn2_model = keras.models.load_model('./models/dnn/trained_2nn_'+str(ntopics)+'.txt')
    dnn3_model = keras.models.load_model('./models/dnn/trained_3nn_'+str(ntopics)+'.txt')

    dictionary = lda_model.id2word

    for raw_text in map(lambda x: rt.raw(x), rt.fileids()):

        bow = dictionary.doc2bow(preprocess_document(raw_text))
        full_bow = np.zeros( (len(dictionary),1) )
        for k, v in dict(bow).items():
            full_bow[int(k)] = int(v)

        td = lda_model[bow]
        full_td_lda = np.zeros((ntopics,1))
        for k, v in dict(td).items():
            full_td_lda[int(k)] = float(v)
        full_td_lda = full_td_lda.transpose()

        full_td_dnn2 = dnn2_model.predict(full_bow.transpose())
        full_td_dnn3 = dnn3_model.predict(full_bow.transpose())
        
        kld2 = np.sum(np.where(full_td_lda != 0, full_td_lda * np.log(full_td_lda / full_td_dnn2), 0))
        kld3 = np.sum(np.where(full_td_lda != 0, full_td_lda * np.log(full_td_lda / full_td_dnn3), 0))
        print("wrt dnn2: ", kld2)
        print("wrt dnn3: ", kld3)
