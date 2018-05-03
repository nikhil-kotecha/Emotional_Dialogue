import tensorflow as tf
import numpy as np
import datetime

from utils import layers


class ClassifyEmotion():
    """
    Classify utterances into emotions
    """

    def __init__(self, num_features=26, num_classes=6, num_hidden=128,
                 dense_hidden=64, lr=1e-4):
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_hidden = num_hidden
        self.dense_hidden = dense_hidden
        self.learning_rate = lr

    def _build_model(self, x, y, seq_len):
        concat_lstm1 = layers.blstm(index=0,
                                    num_hidden=self.num_hidden,
                                    input_x=x,
                                    seq_len=None,
                                    return_all=True)
        concat_lstm1 = tf.expand_dims(concat_lstm1, -1)
        concat_lstm2 = layers.blstm(index=1,
                                    num_hidden=self.num_hidden,
                                    seq_len=None,
                                    input_x=concat_lstm1,
                                    return_all=True)

        with tf.name_scope('dense'):
            dense_0 = tf.layers.dense(concat_lstm2, self.dense_hidden,
                                      activation=tf.nn.tanh)
            print(dense_0.shape)
            logits = tf.layers.dense(dense_0, self.num_classes)
            print(logits.shape)

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                                                           logits=logits))
        return logits, loss

    def _step(self, loss):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    def _accuracy(self, logits, labels):
        with tf.name_scope('accuracy'):
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            return tf.reduce_mean(tf.cast(correct, tf.float32))

    def train(self, train_x, train_y, train_lengths, val_x, val_y, val_lengths,
              batch_size, epochs=20, verbose=True,
              save_path='final_model/emotion_recognition.ckpt',
              load_model=None):
        # Batch data
        batched_data = [train_x[i:i + batch_size] for i in
                        range(0, len(train_x), batch_size)]
        batched_lengths = [train_lengths[i:i + batch_size] for i in
                           range(0, len(train_lengths), batch_size)]
        batched_labels = [train_y[i:i + batch_size] for i in
                          range(0, len(train_y), batch_size)]

        # Batch validation for resource management
        val_batched_data = [val_x[i:i + batch_size] for i in
                            range(0, len(val_x), batch_size)]
        val_batched_lengths = [val_lengths[i:i + batch_size] for i in
                               range(0, len(val_lengths), batch_size)]
        val_batched_labels = [val_y[i:i + batch_size] for i in
                              range(0, len(val_y), batch_size)]

        assert len(batched_data) == len(
            batched_labels), 'Data and labels must have same size.'
        assert len(batched_data) == len(
            batched_lengths), 'Data and lengths must have the same size.'

        print('Created {} batches of size {}'.format(len(batched_data),
                                                     batch_size))
        print('Created Validation {} batches of size {}'.format(
            len(val_batched_data), batch_size))
        print('Data Batch Shape: {} Labels Shape: {} '.format(
            batched_data[0].shape,
            batched_labels[0].shape))

        tf.reset_default_graph()
        tf.set_random_seed(0)

        with tf.name_scope('inputs'):
            x = tf.placeholder(shape=(None, None, self.num_features),
                               dtype=tf.float32)
            y = tf.placeholder(shape=(None, self.num_classes), dtype=tf.float32)
            seq_len = tf.placeholder(shape=(None), dtype=tf.int64)

        logits, loss = self._build_model(x, y, seq_len)
        step = self._step(loss)
        accuracy = self._accuracy(logits, y)

        # Summary ops for loss and accuracy
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        merged = tf.summary.merge_all()

        run = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True

        init = tf.global_variables_initializer()
        with tf.Session(config=config) as sess:
            if load_model:
                saver.restore(sess, load_model)
                print("Model restored.")
            else:
                sess.run(init)

            # Create summary writers for training and testing variables
            train_writer = tf.summary.FileWriter('/tmp/emo/train/' + run,
                                                 sess.graph)
            test_writer = tf.summary.FileWriter('/tmp/emo/test/' + run)

            for epoch in range(epochs):
                batch_losses = []
                for i, batch in enumerate(batched_data):
                    len_batch = batched_lengths[i]
                    y_batch = batched_labels[i]
                    feed_dict = {
                        x: batch,
                        y: y_batch,
                        seq_len: len_batch
                    }
                    _ = sess.run(step, feed_dict=feed_dict)
                    if verbose:
                        err = sess.run(loss, feed_dict=feed_dict)
                        batch_losses.append(err)

                if verbose:
                    summary = sess.run(merged, feed_dict=feed_dict)
                    train_writer.add_summary(summary, epoch)
                    print('Epoch {} mean loss {}'.format(epoch,
                                                         np.mean(batch_losses)))

                if epoch % 5 == 0:
                    val_errors = []
                    for i, val_batch in enumerate(val_batched_data):
                        val_feed_dict = {
                            x: val_batch,
                            y: val_batched_labels[i],
                            seq_len: val_batched_lengths[i]
                        }
                    summary, val_err = sess.run([merged, loss],
                                                feed_dict=val_feed_dict)
                    val_errors.append(val_err)
                    test_writer.add_summary(summary, epoch)
                    print('Epoch {} mean batch val loss {}'.format(epoch,
                                                                   np.mean(
                                                                       val_errors)))

            save_path = saver.save(sess, save_path)

            print('Epoch {} mean batch val loss {}'.format(epoch,
                                                           np.mean(val_errors)))
            val_predictions = []
            for i, val_batch in enumerate(val_batched_data):
                val_feed_dict = {
                    x: val_batch,
                    y: val_batched_labels[i],
                    seq_len: val_batched_lengths[i]
                }
                pred = np.argmax(sess.run(logits, feed_dict=val_feed_dict), 1)
                val_predictions = val_predictions + list(pred)

        return val_predictions

