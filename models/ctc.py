# Bi-Directional LSTMs for Emotion Classification

import tensorflow as tf
import numpy as np

class LSTM_utterance():
    def __init__(self,num_classes, num_features, lr=1e-4):
        self.num_classes = num_classes
        self.num_features = num_features
        self.learning_rate = lr

    def _build_model(self, x, y):
        with tf.variable_scope('lstm1'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(64, state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.LSTMCell(64, state_is_tuple=True)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=x, dtype=tf.float32)
            concat_lstm1 = tf.concat(outputs, 2)
            print(concat_lstm1.shpae)

        with tf.variable_scope('lstm2'):
            cell_fw2 = tf.nn.rnn_cell.LSTMCell(64, state_is_tuple=True)
            cell_bw2 = tf.nn.rnn_cell.LSTMCell(64, state_is_tuple=True)
            outputs_2, states_2 = tf.nn.bidirectional_dynamic_rnn(cell_fw2, cell_bw2, inputs=concat_lstm1, dtype=tf.float32)
            concat_lstm2 = tf.concat(outputs_2, 2)
            concat_lstm2 = tf.transpose(concat_lstm2, [1,0,2])[-1]
            print(concat_lstm2.shape)

        with tf.name_scope('dense'):
            dense_0 = tf.layers.dense(concat_lstm2, 512, activation=tf.nn.tanh)
            print(dense_0.shape)
            logits= tf.layers.dense(dense_0, 6)
            print(logits.shape)

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))

        return logits, loss

    def _step(self, loss):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)


    def train(self, train_x, train_y, val_x, val_y,
              labels, batch_size, epochs=20):
        # Batch data
        batched_data = [train_x[i:i+batch_size] for i in range(0, len(train_x), batch_size)]
        batched_labels = [train_y[i:i+batch_size] for i in range(0, len(train_y), batch_size)]

        with tf.name_scope('inputs'):
            x = tf.placeholder(shape=(None, None, self.num_features), dtype=tf.float32)
            y = tf.placeholder(shape=(None, self.num_classes), dtype=tf.float32)

            logits, loss = self._build_model(x, labels)
            step = self._step(loss)

            tf.reset_default_graph()
            tf.set_random_seed(0)

            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)

                train_losses = []
                val_losses = []
                for epoch in range(epochs):
                    batches_losses = []
                    for i, batch in enumerate(batched_data):
                        labels_batch = batched_labels[batch]
                        feed_dict = {
                            x: batch,
                            y: labels_batch
                        }
                        _, err = sess.run([step, loss], feed_dict=feed_dict)
                        batches_losses.append(err)

                    if epoch % 2 == 0:
                        mean_batch_loss = np.mean(batches_losses)
                        print ('Epoch {} mean loss {}'.format(epoch, mean_batch_loss))
                        val_err = sess.run(loss, feed_dict={x: val_x, y: val_y})
                        print('Epoch {} validation loss'.format(val_err))

            val_predictions = sess.run(logits, feed_dict={x: val_x})
        return train_losses, val_losses, val_predictions





