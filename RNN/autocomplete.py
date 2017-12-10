import numpy as np
import tensorflow as tf
import sys

seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']

idx2char = list(set(''.join(seq_data)))
char2idx = dict({char : idx for idx, char in enumerate(idx2char)})

x_input = []
y_input = []
for word in seq_data:
    tmp_list = [char2idx[char] for char in word[:-1]]
    x_input.append(tmp_list)
    tmp_list = [char2idx[word[-1]]]
    y_input.append(tmp_list)

# hyper parameter
hidden_size = len(idx2char)
batch_size = len(x_input)
num_classes = len(idx2char)

sequence_length = 3
learning_rate = 0.1
epochs = 30

# placeholder for training
x = tf.placeholder(tf.int32, [None, sequence_length])
y = tf.placeholder(tf.int32, [None, 1])
x_one_hot = tf.one_hot(x, num_classes)
y_one_hot = tf.one_hot(y, num_classes)

# Two RNN cell -> fully connected layer -> softmax
# LSTM cell
cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=hidden_size, state_is_tuple=True)

# two LSTM cells
multi_cells = tf.contrib.rnn.MultiRNNCell(
        [cell] * 2, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(
    multi_cells, x_one_hot, dtype=tf.float32)

# fully connected layer
x_for_fc = tf.transpose(outputs, [1, 0, 2])
x_for_fc = x_for_fc[-1]
fc_w = tf.get_variable("fc_w", [hidden_size, num_classes])
fc_b = tf.get_variable("fc_b", [num_classes])
outputs = tf.matmul(x_for_fc, fc_w) + fc_b

# weights = tf.ones([batch_size, num_classes])
loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=y_one_hot)
mean_loss = tf.reduce_mean(loss)
train = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(mean_loss)

# initialization of session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# training
for i in range(epochs):
    mean_loss, output, _= sess.run([loss, outputs, train],
                            feed_dict={x : x_input, y : y_input})
    prediction = np.argmax(output, axis=1)
    pred = [idx2char[idx] for idx in prediction]
    if (i+1) % 10 == 0:
        print('steps :', i+1, 'avg. loss :', np.mean(mean_loss))
        for idx, word in enumerate(seq_data):
            print('input string : %s' %(word[:-1]), 'predicted : %c' %(pred[idx]))
