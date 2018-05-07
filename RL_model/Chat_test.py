# -*- coding: utf-8 -*-

from __future__ import print_function
## standard imports
import re
import os
import time
import sys

sys.path.append("python")
import data_parser ### thanks neural talk
import config

from gensim.models import KeyedVectors ### https://radimrehurek.com/gensim/models/keyedvectors.html
from rl_model import PolicyGradient_chatbot ### the rl model
import tensorflow as tf
import numpy as np

#
# Global Parameters
default_model_path = './model/RL/model-56-3000' ### this is useful to remember
## check the sample input
## this probably has a lot of bearing on where the model goes
testing_data_path = 'sample_input.txt' if len(sys.argv) <= 2 else sys.argv[2]
output_path = 'sample_output_RL.txt' if len(sys.argv) <= 3 else sys.argv[3] ## check the path

word_count_threshold = config.WC_threshold


# Train Parameters
 ### the moment I've been waiting for
dim_wordvec = 300
dim_hidden = 1000 ### double check these numbers here

n_encode_lstm_step = 22 + 1  # one random normal as the first timestep
n_decode_lstm_step = 22 #### standard decode

batch_size = 1 ### Note the batch size her e


## get the vocab correct her
""" Extract only vocabulary from data """


def refine(data): ### start cleaning
    words = re.findall("[a-zA-Z'-]+", data)
    words = ["".join(word.split("'")) for word in words] ## regex expressions review below
    # words = ["".join(word.split("-")) for word in words]
    data = ' '.join(words) ## very good function
    return data


def test(model_path=default_model_path):
    ## make sure everything is working well
    testing_data = open(testing_data_path, 'r').read().split('\n') ### get the path right

    #https: // radimrehurek.com / gensim / models / keyedvectors.html
    word_vector = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True) ### word vector bin -- thanks gensim

    _, ixtoword, bias_init_vector = data_parser.preProBuildWordVocab(word_count_threshold=word_count_threshold)
######
    ###RL model -- policy gradient

    model = PolicyGradient_chatbot(
        dim_wordvec=dim_wordvec, ## wordvec
        n_words=len(ixtoword), ## ixtoword
        dim_hidden=dim_hidden, ## hidden dim
        batch_size=batch_size, ## batch size
        n_encode_lstm_step=n_encode_lstm_step, ## encode
        n_decode_lstm_step=n_decode_lstm_step, ## decode
        bias_init_vector=bias_init_vector)   ## init

    word_vectors, caption_tf, feats = model.build_generator() ## caption

    sess = tf.InteractiveSession() ## boiler plate


### boiler plate
    saver = tf.train.Saver()
    try:
        print('\n=== Use model', model_path, '===\n') ## check model
        saver.restore(sess, model_path)
    except:
        print('\nUse default model\n') ### vanilla
        saver.restore(sess, default_model_path) ## restore model

    with open(output_path, 'w') as out: ##
        generated_sentences = [] ### check the quality of the generated sentences
        bleu_score_avg = [0., 0.] ### this shouldn't work, bleu score is kind of a terrible metric here
        for idx, question in enumerate(testing_data): # enumerate funciton again
            print('question =>', question)

            question = [refine(w) for w in question.lower().split()] ## check the refine funciton, seems to be a bit weird
            question = [word_vector[w] if w in word_vector else np.zeros(dim_wordvec) for w in question]
            question.insert(0, np.random.normal(size=(dim_wordvec,)))  # insert random normal at the first step

            if len(question) > n_encode_lstm_step: ## the len of question needs to be right
                question = question[:n_encode_lstm_step] ## the encode
            else:
                for _ in range(len(question), n_encode_lstm_step): ## o/w all add to np.zeros
                    question.append(np.zeros(dim_wordvec))

            question = np.array([question])  # 1x22x300

            generated_word_index, prob_logit = sess.run([caption_tf, feats['probs']],
                                                        feed_dict={word_vectors: question})
            generated_word_index = np.array(generated_word_index).reshape(batch_size, n_decode_lstm_step)[0] ## array is all off

            ### double check this guy here
            prob_logit = np.array(prob_logit).reshape(batch_size, n_decode_lstm_step, -1)[0] ### this caused a bit of trouble
            # print('generated_word_index.shape', generated_word_index.shape)
            # print('prob_logit.shape', prob_logit.shape)

            # remove <unk> to second high prob. word
            # print('generated_word_index', generated_word_index)
            for i in range(len(generated_word_index)):
                if generated_word_index[i] == 3:
                    sort_prob_logit = sorted(prob_logit[i])
                    # print('max val', sort_prob_logit[-1])
                    # print('second max val', sort_prob_logit[-2])
                    maxindex = np.where(prob_logit[i] == sort_prob_logit[-1])[0][0]
                    secmaxindex = np.where(prob_logit[i] == sort_prob_logit[-2])[0][0]
                    # print('max ind', maxindex, ixtoword[maxindex])
                    # print('second max ind', secmaxindex, ixtoword[secmaxindex])
                    generated_word_index[i] = secmaxindex
            # print('generated_word_index', generated_word_index)

            generated_words = [] ## init list
            for ind in generated_word_index: #### add to list if needd
                generated_words.append(ixtoword[ind])

            # generate sentence
            ## this is so cool
            ### create sentence
            punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1 ### eos
            generated_words = generated_words[:punctuation] ### check the punctuation
            generated_sentence = ' '.join(generated_words) ### goin to new list

            # modify the output sentence

            ### copy and paste from the train part
            generated_sentence = generated_sentence.replace('<bos> ', '') ### ensure grammar right - bos
            generated_sentence = generated_sentence.replace(' <eos>', '') ## eos
            generated_sentence = generated_sentence.replace('--', '') ## em dash
            generated_sentence = generated_sentence.split('  ') ### spaces
            for i in range(len(generated_sentence)): ## throughout the whole range
                generated_sentence[i] = generated_sentence[i].strip() ### check the right function here
                if len(generated_sentence[i]) > 1:
                    generated_sentence[i] = generated_sentence[i][0].upper() + generated_sentence[i][1:] + '.' ### add period
                else:
                    generated_sentence[i] = generated_sentence[i].upper()
            generated_sentence = ' '.join(generated_sentence)
            generated_sentence = generated_sentence.replace(' i ', ' I ') ### get grammar right
            generated_sentence = generated_sentence.replace("i'm", "I'm") ## i'm -> I'm
            generated_sentence = generated_sentence.replace("i'd", "I'd") ## -> I'd
            generated_sentence = generated_sentence.replace("i'll", "I'll") ### -> I'll
            generated_sentence = generated_sentence.replace("i'v", "I'v") ### I've
            generated_sentence = generated_sentence.replace(" - ", "")

            print('generated_sentence =>', generated_sentence)
            out.write(generated_sentence + '\n')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test(model_path=sys.argv[1])
    else:
        test()