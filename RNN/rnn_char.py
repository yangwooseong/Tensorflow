import tensorflow as tf
import numpy as np

# rnn learning helloworld
# helloworl -> elloworld

# set sequence
sequence = "helloworld"
idx2char = [i for i in set(sequence)]
char2idx = {char : idx for idx, char in enumerate(idx2char)}

# hyper parameter
num_classes = len(idx2char)  # num of unique char
sequence_length = len(sequence) - 1  # |helloworl|
hidden_size = len(idx2char)
batch_size = 1
learning_rate = 0.1
epochs = 50

# set train data (before one hot encoding)
x_data = [[char2idx[char] for char in sequence[:-1]]]
y_data = [[char2idx[char] for char in sequence[1:]]]

# placeholder of tensorflow
X = tf.placeholder(
    tf.int32, [None, sequence_length])
Y = tf.placeholder(
    tf.int32, [None, sequence_length])
X_one_hot = tf.one_hot(X, num_classes)

# Banila RNN cell
cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=hidden_size)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(
    cell, X_one_hot, initial_state=initial_state, dtype=tf.float32)

# sequence loss
weights = tf.ones([batch_size, sequence_length])
sequence_loss = tf.contrib.seq2seq.sequence_loss(
    logits=outputs, targets=Y, weights=weights)
train = tf.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(sequence_loss)
prediction = tf.argmax(outputs, axis=2)

sess = tf.Session()

sess.run(tf.global_variables_initializer())
for i in range(epochs):
    l, _ = sess.run([sequence_loss, train], feed_dict={X:x_data, Y:y_data})
    pred = sess.run(prediction, feed_dict={X:x_data})
    pred_str = ''.join([idx2char[idx] for idx in np.squeeze(pred)])
    print(sess.run(outputs, feed_dict={X:x_data}))
    break
    if (i + 1) % 10 ==0 :
        print("%2dstep, loss:%.5f" %(i, l), pred_str)
