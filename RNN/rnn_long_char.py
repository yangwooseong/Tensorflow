import tensorflow as tf
import numpy as np
import sys

# rnn long sentence

# set sequence
sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

idx2char = list(set(sentence))
char2idx = {char : idx for idx, char in enumerate(idx2char)}

# hyper parameter
num_classes = len(idx2char)  # num of unique char
sequence_length = 10 # e.g. 'if you' -> 'f you '
hidden_size = len(idx2char)
batch_size = len(sentence) - sequence_length
# batch_size = len(sentence) - sequence_length
learning_rate = 0.1
epochs = 500

# set train data (before one hot encoding)
x_data = [[ char2idx[char] for char in sentence[i:i+sequence_length] ] for i in range(batch_size)]
y_data = [[ char2idx[char] for char in sentence[i+1:i+1+sequence_length]] for i in range(batch_size)]

# placeholder of tensorflow
X = tf.placeholder(
    tf.int32, [None, sequence_length])
Y = tf.placeholder(
    tf.int32, [None, sequence_length])
X_one_hot = tf.one_hot(X, num_classes)

# Two RNN cell -> fully connected layer -> softmax
# LSTM RNN cell
cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=hidden_size, state_is_tuple=True)

# two LSTM cells
multi_cells = tf.contrib.rnn.MultiRNNCell(
        [cell] * 2, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(
    multi_cells, X_one_hot, dtype=tf.float32)

# fully connected layer
X_for_fc = tf.reshape(outputs, [-1, hidden_size])
outputs = tf.contrib.layers.fully_connected(X_for_fc, num_classes, activation_fn=None)
# shape of outputs : (-1, num_classes)

# reshape for sequence_loss
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

# sequence loss
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
mean_loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(mean_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(epochs):
    loss, output, _ = sess.run([mean_loss, outputs, train], feed_dict={X:x_data, Y:y_data})
    prediction = np.argmax(output, axis=2)
    if (i+1) % 50 == 0:
        print("%2d epoch %.4f loss" %(i+1, loss))

pred_str = ''
for i in range(batch_size):
    if i == 0:
        pred_str += ''.join([idx2char[idx] for idx in prediction[i]])
    else:
        pred_str += idx2char[prediction[i][-1]]
print("Learned sentence is\n",pred_str)
