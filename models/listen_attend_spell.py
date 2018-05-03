import tensorflow as tf
import numpy as np
from utils.layers import blstm, pblstm
from utils.load_data import load_common_voice_data

###########################################################################
# Based on Seq2Seq Tutorial by Minh Thang Luong, Eugene Brevdo, Rui Zhao
# https://www.tensorflow.org/tutorials/seq2seq
# Adapted for reimplementation of Listen, Attend, Spell
# LAS authors: William Chan, Navdeep Jaitly, Quoc V. Le, Oriol Vinyals
###########################################################################

class ListenAttendSpell():
    def __init__(self, enc_num_units=64, max_encode_seq_length=1500,
                 encode_embed_size=26, dec_num_units=128,
                 max_decode_seq_length=100, decode_embed_size = 31,
                 batch_size=32, max_gradient_norm=5, learning_rate=1e-4):
        self.enc_num_units = enc_num_units
        self.max_encode_seq_length = max_encode_seq_length
        self.encode_embed_size = encode_embed_size

        self.dec_num_units = dec_num_units
        self.max_decode_seq_length = max_decode_seq_length
        self.decode_embed_size = decode_embed_size  # ['<sos>', '<eos>', '<pad>'] +
                                                    # 'abcdefghijklmnopqrstuvwxyz \''

        self.batch_size = batch_size
        self.max_gradient_norm = max_gradient_norm
        self.learning_rate = learning_rate


    def _build_train_model(self, enc_input, enc_lengths, dec_input, dec_lengths,
                           max_batch_seq_length):
        """
        Listen, Attend, and Spell model

        Encoder (Listener):
            1 bi-directional LSTM followed by 3 pyramid bi-directional LSTM

            Pyramid BiLSTM reduces sequence length by half in each layer by
                stacking consecutive outputs: see stack() in layers.py

        Attention:
            Used TensorFlow Bahdanau Attention wrapper

        Decoder (Speller):
            Use 1 LSTM layer and the TensorFlow BasicDecoder wrapper for simplicity
        """
        with tf.variable_scope('encoder'):
            blstm0 = blstm(index=0, num_hidden=self.enc_num_units,
                           input_x=enc_input, seq_len=enc_lengths,
                           return_all=False)

            pblstm1 = pblstm(index=1, num_hidden=self.enc_num_units,
                             input_x=blstm0, seq_length=None)

            pblstm2 = pblstm(index=2, num_hidden=self.enc_num_units,
                             input_x=pblstm1, seq_length=None)

            pblstm3, enc_state = pblstm(index=3, num_hidden=self.enc_num_units,
                                        input_x=pblstm2, seq_length=None,
                                        return_state=True)

        with tf.variable_scope('decoder'):
            dec_encoder = tf.get_variable('dec_encoder', [self.decode_embed_size,
                                                          self.decode_embed_size])

            # Embed decoder inputs as 1-hot
            decode_emb_input = tf.nn.embedding_lookup(dec_encoder, dec_input)

            dec_cell = tf.nn.rnn_cell.LSTMCell(self.dec_num_units)

            # Use Bahdanau Attention
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(self.dec_num_units,
                                                                       memory=pblstm3,
                                                                       memory_sequence_length=enc_lengths)
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,
                                                                 attention_mechanism,
                                                                 attention_layer_size=self.dec_num_units / 2)
            dec_out_cell = tf.contrib.rnn.OutputProjectionWrapper(attention_cell,
                                                                  self.decode_embed_size)
            # Seq2seq helper feeds in decoder inputs
            helper = tf.contrib.seq2seq.TrainingHelper(decode_emb_input, dec_lengths)

            decoder = tf.contrib.seq2seq.BasicDecoder(dec_out_cell, helper,
                                                      initial_state=dec_out_cell.zero_state(dtype=tf.float32,
                                                                                            batch_size=self.batch_size))
            outputs = tf.contrib.seq2seq.dynamic_decode(decoder)

            # [batch_size, max_batch_seq_length, dec_embed_size]
            logits = outputs[0].rnn_output

            # Pad dimension 1 of logits to maximum size to make tensors in loss function conformable
            # [batch_size, max_dec_seq_length, dec_embed_size]
            padded_logits = tf.pad(logits, [[0, 0],
                                            [self.max_decode_seq_length - max_batch_seq_length - 1,1],
                                            [0, 0]])
        # argmax predictions
        predictions = outputs[0].sample_id

        # Mask padded inputs
        mask = tf.sequence_mask(dec_lengths, dec_input.shape[-1],
                                dtype=tf.float32)

        # Divide loss by batch size to make loss invariant to batch size
        loss = tf.contrib.seq2seq.sequence_loss(logits=padded_logits,
                                                targets=dec_input,
                                                weights=mask) / self.batch_size

        params = tf.trainable_variables()
        gradients = tf.gradients(loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients,self.max_gradient_norm)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        update_step = optimizer.apply_gradients(
            zip(clipped_gradients, params))

        return predictions, loss, update_step


    def train(self, epochs=20):
        tf.reset_default_graph()
        tf.set_random_seed(0)

        with tf.variable_scope('inputs'):
            # Encoder inputs [batch_size, max_time, embed_size]
            enc_input = tf.placeholder(shape=[self.batch_size, self.max_encode_seq_length, self.encode_embed_size],
                                       dtype=tf.float32)
            enc_lengths = tf.placeholder(shape=[self.batch_size], dtype=tf.int64)

            dec_input = tf.placeholder(shape=[None, self.max_decode_seq_length],
                                       dtype=tf.int32)
            dec_lengths = tf.placeholder(shape=[None], dtype=tf.int32)

            max_batch_seq_length = tf.placeholder(tf.int32)

        predictions, loss, update_step = self._build_model(enc_input, enc_lengths,
                                                           dec_input, dec_lengths,
                                                           max_batch_seq_length)
        # Set Session configuration
        config = tf.ConfigProto()
        config.log_device_placement = True
        config.gpu_options.allow_growth = True

        init = tf.global_variables_initializer()
        with tf.Session(config=config) as sess:
            sess.run(init)
            for epoch in range(epochs):
                # Generator for data batches
                train_batch = load_common_voice_data(use_fbank=True,
                                                     batch_size=self.batch_size,
                                                     load_test=False)
                test_batch = load_common_voice_data(use_fbank=True,
                                                    batch_size=self.batch_size,
                                                    load_test=True)
                num_train_batches = sum(1 for _ in train_batch)
                num_test_batches = sum(1 for _ in test_batch)
                # Batch description
                # item 0 is transcript of characters as integer
                # item 1 is original transcript length
                # item 2 is audio data
                # item 3 is original audio data sequence length

                epoch_losses = []
                for _ in range(num_train_batches-1):
                    sample_batch = next(train_batch)
                    max_batch_length = max(sample_batch[1])
                    # Retrieve next batch and calculate max batch length
                    feed_dict = {
                        enc_input: sample_batch[2],
                        enc_lengths: sample_batch[3],
                        dec_input: sample_batch[0],
                        dec_lengths: sample_batch[1],
                        max_batch_seq_length: max_batch_length
                    }
                    _, err = sess.run([update_step, loss], feed_dict=feed_dict)
                    epoch_losses.append(err)
                print('Epoch {} Mean Train Error {}'.format(epoch, np.mean(epoch_losses)))

                if epoch % 10 == 0:
                    val_epoch_losses = []
                    for _ in range(num_test_batches-1):
                        # Retrieve next batch and calculate max batch length
                        sample_batch = next(test_batch)
                        max_batch_length = max(sample_batch[1])
                        feed_dict = {
                            enc_input: sample_batch[2],
                            enc_lengths: sample_batch[3],
                            dec_input: sample_batch[0],
                            dec_lengths: sample_batch[1],
                            max_batch_seq_length: max_batch_length
                        }
                        val_err = sess.run(loss, feed_dict=feed_dict)
                        val_epoch_losses.append(val_err)
                    print('Epoch {} Mean Validation Error {}'.format(epoch, np.mean(val_epoch_losses)))
