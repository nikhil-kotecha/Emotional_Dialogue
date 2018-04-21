import tensorflow as tf
import numpy as np

def blstm(index, num_hidden, input_x, return_all=False):
    """
    Bidirectional LSTM layer
    Input shape [batch_size, seq_length, embedding_dimension]
    Output shape [batch_size, seq_length, embedding_dimension]

    :return: BLSTM concatenated outputs
    """
    # Get sequence length from input shape
    max_length = input_x.shape[2]

    with tf.variable_scope('blstm_{}'.format(index)):
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                          inputs=input_x,
                                                          dtype=tf.float32,
                                                          sequence_length=max_length)
        concat = tf.concat(outputs, 2)
    if return_all:
        concat = tf.transpose(concat, [1,0,2])[-1]
    print(concat.shape)
    return concat


def pblstm(index, num_hidden, input_x, max_length=2000, return_all=False, return_state=False):
    """
    Pyramidal bidirectional LSTM layer
    Every two conc  atentated output layers are further concatenated

    Input shape [batch_size, seq_length, embedding_dimension]
    Output shape [batch_size, seq_length // 2, embedding_dimension * 2]

    :return:
    """
    with tf.variable_scope('pblstm_{}'.format(index)):
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                          inputs=input_x,
                                                          dtype=tf.float32,
                                                        sequence_length=max_length)
        concat = tf.concat(outputs, 2)
        # Stack every other time step together
    stacked = stack(concat)
    if return_all:
        stacked = stacked[-1]
    if return_state:
        return stacked, states
    else:
        return stacked



def stack(sequence):
    """
    Input tensor of shape [batch_size, seq_length, embedding_dimension]
    Output shape [batch_size, seq_length // 2, embedding_dimension * 2]

    :return: Pyramid stacked sequence
    """
    batch_size = sequence.shape[0]
    seq_length = sequence.shape[1]
    embedding_dim = sequence.shape[2]

    # Pad sequence to even length
    if seq_length % 2 != 0:
        pad = np.zeros((batch_size, 1, embedding_dim))
        sequence = np.concatenate((sequence, pad), axis=1)

    # Split sequences into two by every other sequence
    sequence_even = sequence[:,::2,:]
    sequence_odd = sequence[:,1::2,:]

    stacked_sequence = []
    for i in range(sequence_even.shape[1]):
        stacked = np.concatenate((sequence_even[:, i, :],
                                  sequence_odd[:, i, :]),
                                 axis=1)
        stacked_sequence.append(stacked)
    return np.stack(stacked_sequence)

