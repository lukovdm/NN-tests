import tensorflow as tf
import numpy as np
import csv

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summaries_dir', '/tmp/multiplication_logs', 'Summaries directory')
flags.DEFINE_string('checkpoints_file', 'Multiplication/checkpoints/state.ckpt', 'Checkpoints file')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('batch_size', 100, 'Size of a batch')


def to_float(x):
    return float(x)

to_float = np.vectorize(to_float)


def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read, and
    adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases, layer_name + '/biases')
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.histogram_summary(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, 'activation')
        tf.histogram_summary(layer_name + '/activations', activations)
        return activations


with open("Multiplication/logl.txt") as f:
    data_reader = csv.reader(f, delimiter="\t")
    data = to_float(np.array([list(row) for row in data_reader]))
    np.random.shuffle(data)
    data_set = data[:, :2]
    label_set_tmp = data[:, 2]
    label_set = np.zeros(shape=(label_set_tmp.shape[0], 100))
    for i in range(label_set_tmp.shape[0]):
        label_set[i, int(label_set_tmp[i])-1] = 1
        if i % int(label_set_tmp.shape[0] / 10) == 0:
            print(str(int(i/label_set_tmp.shape[0] * 100)) + "%")
    del data
    del label_set_tmp

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, (FLAGS.batch_size, 2))
    y_ = tf.placeholder(tf.float32, (FLAGS.batch_size, 100))

hidden1 = nn_layer(x, 2, 100, "hidden_1")
hidden2 = nn_layer(hidden1, 100, 100, "hidden_2")
output = nn_layer(hidden2, 100, 100, "output", act=tf.nn.softmax)

with tf.name_scope("cost"):
    diff = y_ * tf.log(output)
    with tf.name_scope('total'):
        cost = -tf.reduce_mean(diff)
    tf.scalar_summary('cost', cost)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(
        FLAGS.learning_rate).minimize(cost)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.scalar_summary('accuracy', accuracy)

init_op = tf.initialize_all_variables()

with tf.Session() as sess:

    # summaries #
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph, flush_secs=2)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test', flush_secs=2)

    # saving #
    saver = tf.train.Saver()

    # run model #
    sess.run(init_op)
    print("starting training")
    for i in range(0, int(label_set.shape[0]/FLAGS.batch_size)*FLAGS.batch_size, 100):
        batch_x = data_set[i:i+FLAGS.batch_size]
        batch_y = label_set[i:i+FLAGS.batch_size]
        if i % 20 * FLAGS.batch_size == 0:
            # test network #
            summary, acc = sess.run([merged, accuracy], feed_dict={x: batch_x, y_: batch_y})
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
            if i % 100 * FLAGS.batch_size == 0:
                saver.save(sess, FLAGS.checkpoints_file)
        else:
            # train network #
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch_x, y_: batch_y})
            train_writer.add_summary(summary, i)
