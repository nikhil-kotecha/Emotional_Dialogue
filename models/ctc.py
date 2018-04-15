import tensorflow as tf
from random import shuffle

class LSTMctc():
    def __init__(self,num_classes, num_features, lr=1e-4):
        self.num_classes = num_classes
        self.num_features = num_features
        self.learning_rate = lr

    def _build_model(self, x, y):
        with tf.variable_scope('lstm1'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(64, state_is_tuple=True)
            cell_bw = tf.nn.rnn_cell.LSTMCell(64, state_is_tuple=True)
            outputs_1, states_1 = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=x)
            concat_lstm1 = tf.concat(outputs_1, 2)

        with tf.variable_scope('lstm2'):
            cell_fw2 = tf.nn.rnn_cell.LSTMCell(64, state_is_tuple=True)
            cell_bw2 = tf.nn.rnn_cell.LSTMCell(64, state_is_tuple=True)
            outputs_2, states_2 = tf.nn.bidirectional_dynamic_rnn(cell_fw2, cell_bw2, inputs=concat_lstm1)
            concat_lstm2 = tf.concat(outputs_2, 2)

        with tf.name_scope('dense'):
            dense_0 = tf.layers.dense(concat_lstm2, 512, activation=tf.nn.tanh)
            dense_1 = tf.layers.dense(dense_0, 4, activation=tf.nn.tanh)
            logits= tf.layers.dense(dense_1, self.num_classes)

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.ctc_loss(labels=y, inputs=logits, sequence_length=self.batch_size))

        return logits, loss


    def _optimizer(self, loss):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)


    def train(self, data, labels, epochs=20):
        # Batch data
        batched_data = [data[i:i+self.batch_size] for i in range(0, len(data), self.batch_size)]
        num_batches = len(batched_data)

        # Convert Y to one-hot
        labels = tf.one_hot(labels)
        batched_labels = [labels[i:i+self.batch_size] for i in range(0, len(labels), self.batch_size)]

        with tf.name_scope('inputs'):
            x = tf.placeholder(shape=[None, None, self.num_features], dtype=tf.float32)
            y = tf.placeholder(shape=[None, self.num_classes], dtype=tf.float32)
            batch_size = tf.placeholder(shape=[None], dtype=tf.int64)

            logits, loss = self._build_model(x, labels, batch_size)
            optimizer = self._optimizer(loss)

            tf.reset_default_graph()
            tf.set_random_seed(0)

            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)

                for epoch in range(epochs):
                    for i, batch in enumerate(batched_data):
                        labels_batch = batched_labels[batch]
                        feed_dict = {
                            x: batch,
                            y: labels_batch
                        }
                        shuffle(batch)
                        sess.run([loss, optimizer], feed_dict=feed_dict)





