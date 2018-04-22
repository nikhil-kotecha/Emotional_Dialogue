import tensorflow as tf
from utils.layers import blstm, pblstm

###########################################################################
# Based on Seq2Seq Tutorial by Minh Thang Luong, Eugene Brevdo, Rui Zhao
# https://www.tensorflow.org/tutorials/seq2seq
# Adapted for reimplementation of Listen, Attend Spell
# LAS authors: William Chan, Navdeep Jaitly, Quoc V. Le, Oriol Vinyals
###########################################################################

class ListenAttendSpell():
    def __init__(self, max_input_seq_length=2000, max_output_seq_length=100,
                 input_vocab_size=26, output_vocab_size=28,
                 max_gradient_norm=5, learning_rate=1e-4,):
        self.max_input_seq_length = max_input_seq_length
        self.max_output_seq_length = max_output_seq_length
        self.input_vocab_size = input_vocab_size  # 26 fbank or mfcc features
        self.output_vocab_size = output_vocab_size  # 26 lower case letters, space, apostrophe
        self.max_gradient_norm = max_gradient_norm
        self.learning_rate = learning_rate


    def _build_train_model(self, enc_input, dec_input):
        """
        Listen, Attend, and Spell model
        """
        with tf.name_scope('listen'):
            """
            Encoder is a pyramidal bi-directional LSTM to encode 
            audio features into high level features 

            Each layer reduces the time resolution by a factor of 2   
            
            Encoder input has shape [batch_size, max_input_seq_length, input_vocab_size] 
            Deocder input has shape [batch_size, max_output_seq_length, output_vocab_size] 
            """
            blstm0 = blstm(index=0, num_hidden=256, input_x=enc_input, return_all=True)
            pblstm1= pblstm(index=1, num_hidden=256, input_x=blstm0, return_all=True)
            pblstm2 = pblstm(index=2, num_hidden=256, input_x=pblstm1, return_all=True)
            pblstm3, enc_state = pblstm(index=3, num_hidden=256, input_x=pblstm2,
                             return_all=True, return_state=True )

        with tf.name_scope('attend'):
            # Attention states are [batch_size, seq_length, embed_dimension]
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=256, memory=pblstm3, memory_seq_length=2000, normalize=True)
            cell = tf.contrib.rnn.LSTMCell(num_units=256)
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell, attention_mechanism, attention_layer_size=256//2)
            out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                attention_cell, 26, reuse=reuse)

        with tf.name_scope('spell'):
             with tf.variable_scope('decoder'):
                 decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(512)

                 # The paper uses a 10% scheduled sampling rate
                 # Scheduled sampling is when the decoder reads from the predicted output
                 # rather than the ground truth
                 helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(
                     out_cell, sampling_probability=0.1, name='training_helper'
                 )
                 decoder = tf.contrib.seq2seq.BasicDecoder(
                     decoder_cell, helper, enc_state)
                 outputs,_ = tf.contrib.seq2seq.dynamic_decode(decoder)
                 logits = outputs.rnn_output


        with tf.name_scope('loss'):
            """
            Character level cross entropy loss
            """
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                      labels=dec_input)
            loss = tf.reduce_sum(crossent)

        return logits, loss


    def _step(self, loss):
        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _  = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        return optimizer.apply_gradients(zip(clipped_gradients, params))


    def train(self, data, label, batch_size=128, epochs=20):
        batched_data = [data[i:i+batch_size] for i in range(0, data.shape[0], batch_size)]
        batched_labels = [label[i:i+batch_size] for i in range(0, label.shape[0], batch_size)]
        num_batches = len(batched_data)

        with tf.variable_scope('inputs'):
            encoder_inputs = tf.placeholder(shape=(None, None, self.input_vocab_size))
            decoder_inputs = tf.placeholder(shape=(None, None, self.output_vocab_size))

        logits, loss = self._build_train_model(encoder_inputs, decoder_inputs)
        step = self._step(loss)

        for epoch in range(epochs):
            batches_losses = []
            for batch in range(num_batches-1):
                batch_enc = batched_data[batch]
                batch_dec = batched_labels[batch]
                feed_dict = {
                    encoder_inputs: batch_enc,
                    decoder_inputs: batch_dec
                }
                _, err = sess.run([step, loss], feed_dict=feed_dict)
                batches_losses.append(err)
                if epoch % 2 == 0:
                    mean_batch_loss = np.mean(batches_losses)
                    print ('Epoch {} mean loss {}'.format(epoch, mean_batch_loss))

                if epoch % 10 == 0:
                    val_err = sess.run([merged, loss], feed_dict={x: batched_data[-1],
                                                                  y: batched_data[-1]})
                    print('Epoch {} validation loss'.format(val_err))
                    









