# coding=utf-8

#import the necessary elements
##tensorflow + numpy
import tensorflow as tf
import numpy as np

class Seq2Seq_chatbot():
    "Used this link as a guide: https://www.tensorflow.org/tutorials/seq2seq"
    def __init__(self, dim_wordvec, n_words, dim_hidden, batch_size, n_encode_lstm_step, n_decode_lstm_step, bias_init_vector=None, lr=0.0001):
        self.dim_wordvec = dim_wordvec #wordvec important to have this specified
        self.dim_hidden = dim_hidden #number of hidden layers
        self.batch_size = batch_size #batch size, more training examples. Could try the dynamic rnn for better
        self.n_words = n_words #Could do Dynamic RNN for faster comp to get the exact source sentence len but need that information specified
        self.n_encode_lstm_step = n_encode_lstm_step  ### need encoder
        self.n_decode_lstm_step = n_decode_lstm_step ###need decoder
        self.lr = lr

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb') ## get things proper

        self.lstm1 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=False) ###use the Basic LSTM cell - could use the attention wrapper here
        self.lstm2 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=False) #### once again the attention wrapper could be used for better performance.

        self.encode_vector_W = tf.Variable(tf.random_uniform([dim_wordvec, dim_hidden], -0.1, 0.1), name='encode_vector_W')  ###encoding W
        self.encode_vector_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_vector_b') ##### encoding B

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W') ### pay attention to embeddings -- http://ruder.io/word-embeddings-1/index.html
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')  ##add some zeros if need be

    def build_model(self):
        word_vectors = tf.placeholder(tf.float32, [self.batch_size, self.n_encode_lstm_step, self.dim_wordvec]) ##really like these placeholder elements

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_decode_lstm_step+1])  #### used the neuraltalk2 link to help here.
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_decode_lstm_step+1]) ### That karpathy guy is everywhere

        word_vectors_flat = tf.reshape(word_vectors, [-1, self.dim_wordvec])
        wordvec_emb = tf.nn.xw_plus_b(word_vectors_flat, self.encode_vector_W, self.encode_vector_b ) # (batch_size*n_encode_lstm_step, dim_hidden)
        wordvec_emb = tf.reshape(wordvec_emb, [self.batch_size, self.n_encode_lstm_step, self.dim_hidden]) ###pay att'n to dimensions ^ are reshaped

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size]) ###zeros all around
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size]) ## more zero
        padding = tf.zeros([self.batch_size, self.dim_hidden]) ## more zero

        probs = [] ##init
        entropies = [] ## init
        loss = 0.0 ##init

        ##### Encoding Stage --- LSTM section
        for i in range(0, self.n_encode_lstm_step): ### watch out for for loops
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"): ###LSTM - output1, state1 which feed into LSTM 2
                output1, state1 = self.lstm1(wordvec_emb[:, i, :], state1)

            with tf.variable_scope("LSTM2"): ### concatenate, output2, state2
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)  ##### re-add the padding.

        ##### Decoding Stage
        for i in range(0, self.n_decode_lstm_step): ### very similar to encoding stage
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i]) ### look out for embeds

            tf.get_variable_scope().reuse_variables() ### enjoy this function a lot

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1) ### symmetrical to encoding, output1, state1. Notice padding

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2) #### notice concatenation with current embeddings

            labels = tf.expand_dims(caption[:, i+1], 1) ## check dimensions - those colons always throw me
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) ### batch size, careful after changing variable name
            concated = tf.concat([indices, labels], 1) ## concatenate

            ####https://stackoverflow.com/questions/42127505/tensorflow-dense-to-sparse
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0) #### sparse to dense

            ######https://github.com/tensorflow/tensorflow/issues/1434
            ####should make switch tf.matmul(x,w) + b
            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b) ### this method isn't well documented. If time change over to tf.matmul
            ##cross entropy
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels) ### one hot labels here
            cross_entropy = cross_entropy * caption_mask[:, i] ### check dims for caption mask
            entropies.append(cross_entropy) ## add to the entropy list
            probs.append(logit_words)

            current_loss = tf.reduce_sum(cross_entropy)/self.batch_size ### normalize by batch size per the tensorflow
            loss = loss + current_loss ## update loss


        ##watch out for ADAM. There was thing about clipping gradients
        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            train_op = tf.train.AdamOptimizer(self.lr).minimize(loss) ### check on gradient clipping

        inter_value = {
            'probs': probs,
            'entropies': entropies
        }

        return train_op, loss, word_vectors, caption, caption_mask, inter_value ### return makes it all worthwhile!

    def build_generator(self):
        word_vectors = tf.placeholder(tf.float32, [1, self.n_encode_lstm_step, self.dim_wordvec]) ## do the word vecs here
        # watch out for dims here
        word_vectors_flat = tf.reshape(word_vectors, [-1, self.dim_wordvec]) ## notice the -1, forgot the first time
        #watch matmul
        wordvec_emb = tf.nn.xw_plus_b(word_vectors_flat, self.encode_vector_W, self.encode_vector_b) ### switch to tf.matmul

        wordvec_emb = tf.reshape(wordvec_emb, [1, self.n_encode_lstm_step, self.dim_hidden]) ## encode lstm

        state1 = tf.zeros([1, self.lstm1.state_size]) ### state1
        state2 = tf.zeros([1, self.lstm2.state_size]) ### state2
        padding = tf.zeros([1, self.dim_hidden]) ####watch padding

        generated_words = [] # init

        probs = [] ## init
        embeds = [] ## init

        for i in range(0, self.n_encode_lstm_step): ## careful on for loop. Here for counting
            if i > 0:
                tf.get_variable_scope().reuse_variables() ### reuse

###LSTMS 1 / 2
            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(wordvec_emb[:, i, :], state1) ### do LSTM 1

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2) ### LSTM2

        for i in range(0, self.n_decode_lstm_step):
            tf.get_variable_scope().reuse_variables() ### for loop

            if i == 0: ### embedding lookup. Always have to double check
                with tf.device('/cpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1) ## padding

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2) ### LSTM 2

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b) ### check out tf.matmul
            max_prob_index = tf.argmax(logit_words, 1)[0] ### argmax. Accidentally wrote max earlier
            generated_words.append(max_prob_index) ### check out gen words
            probs.append(logit_words) ### logit

            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index) ### no gpu for you
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return word_vectors, generated_words, probs, embeds ### feels good