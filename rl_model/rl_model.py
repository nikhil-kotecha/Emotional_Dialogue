# coding=utf-8

import tensorflow as tf
import numpy as np

## Reference Resource: https://github.com/liuyuemaicha/Deep-Reinforcement-Learning-for-Dialogue-Generation-in-tensorflow

class PolicyGradient_chatbot():
    def __init__(self, dim_wordvec, n_words, dim_hidden, batch_size, n_encode_lstm_step, n_decode_lstm_step, bias_init_vector=None, lr=0.0001):
       #c #wordvec important to have this specified
       # self.dim_hidden = dim_hidden #number of hidden layers
        #self.batch_size = batch_size #batch size, more training examples. Could try the dynamic rnn for better
        #self.n_words = n_words #Could do Dynamic RNN for faster comp to get the exact source sentence len but need that information specified
        #self.n_encode_lstm_step = n_encode_lstm_step  ### need encoder
        #self.n_decode_lstm_step = n_decode_lstm_step ###need decoder


        self.dim_wordvec = dim_wordvec ## very similar model to seq2seq before, just a bit more
        self.dim_hidden = dim_hidden # make sure get vars right
        self.batch_size = batch_size ## not sure what batch size to use
        self.n_words = n_words ## all good
        self.n_encode_lstm_step = n_encode_lstm_step ## encode
        self.n_decode_lstm_step = n_decode_lstm_step ## decode
        self.lr = lr

        with tf.device("/cpu:0"): ### yay cpu
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

######### https://www.tensorflow.org/tutorials/seq2seq#attention_wrapper_api
        self.lstm1 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=False) #### maybe add the attention wrapper
        self.lstm2 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=False) #### watch the basic lstm

        self.encode_vector_W = tf.Variable(tf.random_uniform([dim_wordvec, dim_hidden], -0.1, 0.1), name='encode_vector_W') ## encoding on W
        self.encode_vector_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_vector_b') ### watch b

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W') ## get embeddings right
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b') ### np. float 32. had at 64 for some reason
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b') ## cued from seq2seq

    def build_model(self):
        #### very similar to before
        word_vectors = tf.placeholder(tf.float32, [self.batch_size, self.n_encode_lstm_step, self.dim_wordvec])

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_decode_lstm_step+1]) ### caption the variables
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_decode_lstm_step+1]) ### look at next step, pay attn to the dims

        word_vectors_flat = tf.reshape(word_vectors, [-1, self.dim_wordvec]) ### flattened
        ### tf. matmul would be better
        wordvec_emb = tf.nn.xw_plus_b(word_vectors_flat, self.encode_vector_W, self.encode_vector_b ) # (batch_size*n_encode_lstm_step, dim_hidden)
        wordvec_emb = tf.reshape(wordvec_emb, [self.batch_size, self.n_encode_lstm_step, self.dim_hidden]) ## reshpae vec

        # now we're getting into some differences
        reward = tf.placeholder(tf.float32, [self.batch_size, self.n_decode_lstm_step]) ### reward function init

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size]) ## init with zero
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])  ## init with zero
        padding = tf.zeros([self.batch_size, self.dim_hidden])  ## init with zero

        entropies = [] ### init
        loss = 0. ### init
        pg_loss = 0.  # policy gradient loss

        ##### Encoding Stage
        for i in range(0, self.n_encode_lstm_step): ### for loop
            if i > 0:
                tf.get_variable_scope().reuse_variables() ### reuse variables is the best

            with tf.variable_scope("LSTM1"): ## check lstm 1 out
                output1, state1 = self.lstm1(wordvec_emb[:, i, :], state1) ## get embeds right
                # states.append(state1) ## watch this, something wonky here

            with tf.variable_scope("LSTM2"): ### lstm 2
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2) ### concat and get right

        ###Decoding Stage
        for i in range(0, self.n_decode_lstm_step): ## for loop
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i]) ## look at caption

            tf.get_variable_scope().reuse_variables() ## reuse

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1) ### padding is imp, forgot that earlier

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2) ###concat

            labels = tf.expand_dims(caption[:, i+1], 1) ### expand dims
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) ### forgot the outer one, careful
            concated = tf.concat([indices, labels], 1) ## check index
            ## stack it right
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0) ## look for this



            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)### tf.matmul might be worth swtiching too

            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels) ### softmax, sparse?
            cross_entropy = cross_entropy * caption_mask[:, i]
            entropies.append(cross_entropy)
            pg_cross_entropy = cross_entropy * reward[:, i] ### pg cross entropy --- notice reward

            pg_current_loss = tf.reduce_sum(pg_cross_entropy) / self.batch_size ### normalize by batch size
            pg_loss = pg_loss + pg_current_loss

        with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            train_op = tf.train.AdamOptimizer(self.lr).minimize(pg_loss) ### ADAM optimizer. Stay consistent with before

        input_tensors = {
            'word_vectors': word_vectors,
            'caption': caption,
            'caption_mask': caption_mask,
            'reward': reward
        }

        feats = {
            'entropies': entropies
        }

        return train_op, pg_loss, input_tensors, feats

    def build_generator(self):
        ##tf. float 32
        word_vectors = tf.placeholder(tf.float32, [self.batch_size, self.n_encode_lstm_step, self.dim_wordvec])


######## Very similar to before
        word_vectors_flat = tf.reshape(word_vectors, [-1, self.dim_wordvec])
        wordvec_emb = tf.nn.xw_plus_b(word_vectors_flat, self.encode_vector_W, self.encode_vector_b)
        wordvec_emb = tf.reshape(wordvec_emb, [self.batch_size, self.n_encode_lstm_step, self.dim_hidden])

        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        padding = tf.zeros([self.batch_size, self.dim_hidden]) ## check the padding

        generated_words = []

        probs = []
        embeds = []
        states = []

        for i in range(0, self.n_encode_lstm_step): # for loop
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(wordvec_emb[:, i, :], state1)
                states.append(state1) ## get errything right

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

        for i in range(0, self.n_decode_lstm_step):
            tf.get_variable_scope().reuse_variables() ## decode

            if i == 0:
                # <bos>

                ### pay attn ^
                with tf.device('/cpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([self.batch_size], dtype=tf.int64))

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1) ## padding

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b) ## tf. matmul
            max_prob_index = tf.argmax(logit_words, 1) ### argmax, copied and pasted wrong thing earlier
            generated_words.append(max_prob_index)
            probs.append(logit_words) ### watch logit here

            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index) ## double check Wemb  is imp

            embeds.append(current_embed)

        feats = {
            'probs': probs,
            'embeds': embeds,
            'states': states
        }

        return word_vectors, generated_words, feats