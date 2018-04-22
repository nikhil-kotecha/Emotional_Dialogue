import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from utils import layers


class ClassifyEmotion():
    """
    Classify utterances into emotions
    """
    def __init__(self, num_features=26, num_classes=6, lr=1e-4):
        self.num_classes = num_classes
        self.num_features = num_features
        self.learning_rate = lr

    def _build_model(self, x, y):
        concat_lstm1 = layers.blstm(index=0,
                                    num_hidden=128,
                                    input_x=x,
                                    return_all=True)
        concat_lstm2 = layers.blstm(index=1,
                                    num_hidden=128,
                                    input_x=concat_lstm1,
                                    return_all=False)

        with tf.name_scope('dense'):
            dense_0 = tf.layers.dense(concat_lstm2, 512, activation=tf.nn.tanh)
            print(dense_0.shape)
            logits= tf.layers.dense(dense_0, self.num_classes)
            print(logits.shape)

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                                                                             logits=logits))
        return logits, loss

    def _step(self, loss):
        return tf.train.AdamOptimizer(self.learning_rate).minimize(loss)

    def _accuracy(self, logits, labels):
        with tf.name_scope('accuracy'):
            correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            return tf.reduce_mean(tf.cast(correct, tf.float32))


    def train(self, data, label, batch_size, epochs=20):
        # Convert Y to one hot
        int_labels = LabelEncoder().fit_transform(label)
        int_labels = int_labels.reshape(len(int_labels), 1)
        labels = OneHotEncoder().fit_transform(int_labels).toarray()

        # Split into train and test
        train_x, val_x, train_y, val_y = train_test_split(data, labels,
                                                          test_size=0.2)
        # Batch data
        batched_data = [train_x[i:i+batch_size] for i in range(0, len(train_x), batch_size)]
        batched_labels = [train_y[i:i + batch_size] for i in range(0, len(train_y), batch_size)]

        assert len(batched_data) == len(batched_labels), 'Data and labels must have same size.'

        print('Created {} batches of size {}'.format(len(batched_data), batch_size))
        print('Data Batch Shape: {} Labels Shape: {} '.format(batched_data[0].shape,
                                                              batched_labels[0].shape))

        tf.reset_default_graph()
        tf.set_random_seed(0)

        with tf.name_scope('inputs'):
            x = tf.placeholder(shape=(None, None, self.num_features), dtype=tf.float32)
            y = tf.placeholder(shape=(None, self.num_classes), dtype=tf.float32)

        logits, loss = self._build_model(x, y)
        step = self._step(loss)
        accuracy = self._accuracy(logits, y)

        # Summary ops for loss and accuracy
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        merged = tf.summary.merge_all()

        # Create summary writers for training and testing variables
        train_writer = tf.summary.FileWriter('/tmp/emo/train/')
        test_writer = tf.summary.FileWriter('tmp/emo/test/')

        saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            train_losses = []
            val_losses = []
            for epoch in range(epochs):
                batches_losses = []
                for i, batch in enumerate(batched_data):
                    y_batch = batched_labels[i]
                    feed_dict = {
                        x: batch,
                        y: y_batch
                    }

                    _, err = sess.run([step, loss], feed_dict=feed_dict)
                    batches_losses.append(err)

                if epoch % 2 == 0:
                    summary = sess.run(merged, feed_dict=feed_dict)
                    train_writer.add_summary(summary, epoch )
                    mean_batch_loss = np.mean(batches_losses)
                    print ('Epoch {} mean loss {}'.format(epoch, mean_batch_loss))

                if epoch % 10 == 0:
                    summary, val_err = sess.run([merged, loss], feed_dict={x: val_x, y: val_y})
                    test_writer.add_summary(summary, epoch)
                    print('Epoch {} validation loss'.format(val_err))

            val_predictions = sess.run(logits, feed_dict={x: val_x})

        save_path = saver.save(sess, 'final_model/emotion_recognition.ckpt')
        return val_predictions

