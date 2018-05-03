import tensorflow as tf


def blstm(index, num_hidden, input_x, seq_len, return_all=False):
    """
    Bidirectional LSTM layer
    Input shape [batch_size, seq_length, embedding_dimension]
    Output shape [batch_size, seq_length, embedding_dimension]

    :return: BLSTM concatenated outputs
    """

    with tf.variable_scope('blstm_{}'.format(index)):
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                          inputs=input_x,
                                                          dtype=tf.float32,
                                                          sequence_length=seq_len)
        concat = tf.concat(outputs, 2)
    if return_all:
        concat = tf.transpose(concat, [1, 0, 2])[-1]
    print('BLSTM-{}'.format(index), concat.shape)
    return concat


def pblstm(index, num_hidden, input_x, seq_length=None, return_state=False):
    """
    Pyramidal bidirectional LSTM layer
    Every two concatentated output layers are further concatenated

    Input shape [batch_size, seq_length, embedding_dimension]
    Output shape [batch_size, seq_length // 2, embedding_dimension * 2]

    :return: if return_state: stacked concatenated outputs and concatenated states
             else stacked concatenated outputs only
    """
    with tf.variable_scope('pblstm_{}'.format(index)):
        cell_fw = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
        cell_bw = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                          inputs=input_x,
                                                          dtype=tf.float32,
                                                          sequence_length=seq_length)
    if return_state:
         concat_outputs = tf.concat(outputs, 2)
         stacked = stack(concat_outputs)
         print('P-BiLSTM-{}'.format(index), stacked.shape)
         concat_states = tf.concat(states, 2)
         print('P-BiLSTM-{}-States'.format(index), concat_states.shape)
         return stacked, concat_states
    else:
         # Stack every other time step together
         concat = tf.concat(outputs, 2)
         stacked = stack(concat)
         print('P-BiLSTM-{}'.format(index), stacked.shape)
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
        pad = tf.zeros((batch_size, 1, embedding_dim), dtype=tf.float32)
        sequence = tf.concat((sequence, pad), axis=1)

    # Split sequences into two by every other sequence
    even = sequence[:,::2, :]
    odd = sequence[:,1::2, :]

    return tf.concat((even, odd), axis=2)

