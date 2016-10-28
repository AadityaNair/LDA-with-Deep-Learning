#!/bin/bash

if [[ -z $NUM_TOPICS ]]; then
	export WORKERS=7
fi
if [[ -z $NUM_TOPICS ]]; then
	export NUM_TOPICS=20
fi

if [[ -n $GENERATE_INPUTS ]]; then
	python gen_input.py  # ensure that there is an lda somewhere to give the script a dictionary object
fi

export LDA_MODEL="./models/lda/trained_lda_${PASSES}_${NUM_TOPICS}.txt"
export _2NN_MODEL="./models/dnn/trained_2nn_${NUM_TOPICS}.txt"
export _3NN_MODEL="./models/dnn/trained_3nn_${NUM_TOPICS}.txt"

python lda.py && python gen_lda_output.py

python svm.py > lda_accuracy_${NUM_TOPICS}

export DNN_MODEL=$_2NN_MODEL

python 2nn.py && python gen_dnn_output.py

python svm.py > 2nn_accuracy_${NUM_TOPICS}

export DNN_MODEL=$_3NN_MODEL

python 3nn.py && python gen_dnn_output.py

python svm.py > 3nn_accuracy_${NUM_TOPICS}
