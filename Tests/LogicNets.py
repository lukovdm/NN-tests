import tensorflow as tf
import numpy as np

NUM_HIDDEN = 2
NUM_ITERATIONS = 1000


def create_x_input(i):
    if i % 4 == 0:
        return np.array([0, 0])
    elif i % 4 == 1:
        return np.array([1, 0])
    elif i % 4 == 2:
        return np.array([0, 1])
    elif i % 4 == 3:
        return np.array([1, 1])


def create_y_input(i):
    if i % 4 == 0:
        return np.array([0])
    elif i % 4 == 1:
        return np.array([1])
    elif i % 4 == 2:
        return np.array([1])
    elif i % 4 == 3:
        return np.array([0])

trX = np.array([create_x_input(i) for i in range(NUM_ITERATIONS)])
trY = np.array([create_y_input(j) for j in range(NUM_ITERATIONS)])

teX = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
teY = np.array([[0], [1], [1], [0]])

X = tf.placeholder(tf.float32, shape=(None, 2))
Y = tf.placeholder(tf.float32, shape=(None, 1))


def init_weights(shape, name):
    weights = tf.Variable(tf.random_normal(shape, stddev=10), name=name)
    tf.histogram_summary(name, weights)
    return weights

w_h = init_weights([2, NUM_HIDDEN], 'hidden-l1/weights')
b_h = init_weights([1, 1], 'hidden-l1/biases')
w_o = init_weights([NUM_HIDDEN, 1], 'output/weights')
b_o = init_weights([1, 1], 'output/biases')

h = tf.nn.sigmoid(tf.matmul(X, w_h) + b_h, name='hidden-l1')
o = tf.nn.sigmoid(tf.matmul(h, w_o) + b_o, name='output')

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(o, Y), name='loss-function')
optimizer = tf.train.AdamOptimizer()
tf.scalar_summary("loss", loss)

train_op = optimizer.minimize(loss)
predict_op = tf.round(o)
summary_op = tf.merge_all_summaries()


def feed_dict(X_, Y_, epoch_):
    return {X_: np.transpose(X_[epoch_, ]), Y_: np.transpose(Y_[epoch_, ])}

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    summary_writer = tf.train.SummaryWriter('/tmp/xor_log', sess.graph)
    p = np.random.permutation(range(len(trX)))
    trX, trY = trX[p], trY[p]

    for epoch in range(NUM_ITERATIONS):

        sess.run(train_op, feed_dict=feed_dict(trX, trY, epoch))
        summary_str = sess.run(summary_op, feed_dict=feed_dict(trX, trY, epoch))
        summary_writer.add_summary(summary_str, epoch)

        print(epoch, np.reshape(sess.run(predict_op, feed_dict={X: teX, Y: teY}), 4))
