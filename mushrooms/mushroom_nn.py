import tensorflow as tf
import numpy as np

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)

ascii_diff = ord('a') - 1
edible_code = ord('e') - ascii_diff
train_size = 8000
train_width = 5

#   정수값으로 계산을 하게 되면 수의 크기가 python에서 다루기 너무 커지므로
#   데이터 값의 크기를 줄임
normalize = 0.1

#   edible / poison mushroom
nb_classes = 2

learning_rate = 1e-6

raw_data = np.loadtxt('./mushrooms.csv', delimiter=',', dtype=np.character)
data = np.empty((len(raw_data), len(raw_data[0])))

for i in range(len(raw_data)):
    for j in range(len(raw_data[i])):
        data[i][j] = int(ord(raw_data[i][j]) - ascii_diff)

x_data = np.empty((train_size, len(data[0]) - 1))
y_data = np.empty((train_size, 2))

x_test = np.empty((len(data) - train_size, len(data[0]) - 1))
y_test = np.empty((len(data) - train_size, 2))

# y_data = [edible, poison]
for i in range(train_size):
    #   데이터의 값이 너무 크므로 일정 비율로 줄여준다
    x_data[i] = data[i][1:]# * normalize
    
    # is edible?
    if data[i][0] == edible_code:
        y_data[i] = [1, 0]
    else:
        y_data[i] = [0, 1]

for i in range(len(data) - train_size):
    #   데이터의 값이 너무 크므로 일정 비율로 줄여준다
    x_test[i] = data[i + train_size][1:]# * normalize
    
    # is edible?
    if data[i + train_size][0] == edible_code:
        y_test[i] = [1, 0]
    else:
        y_test[i] = [0, 1]

x_data = MinMaxScaler(x_data)
x_test = MinMaxScaler(x_test)

X = tf.placeholder(tf.float32, [None, len(x_data[0])], name='x-input')
Y = tf.placeholder(tf.float32, [None, nb_classes], name='y-input')

with tf.name_scope('layer1') as scope:
    W1 = tf.get_variable("W1", shape=[len(x_data[0]), train_width], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([train_width]), name='bias1')
    layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)

    w1_hist = tf.summary.histogram("weights1", W1)
    b1_hist = tf.summary.histogram("biases1", b1)
    layer1_hist = tf.summary.histogram("layer1", layer1)

with tf.name_scope('layer2') as scope:
    W2 = tf.get_variable("W2", shape=[train_width, train_width], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([train_width]), name='bias2')
    layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

    w2_hist = tf.summary.histogram("weights2", W2)
    b2_hist = tf.summary.histogram("biases2", b2)
    layer2_hist = tf.summary.histogram("layer2", layer2)

with tf.name_scope('layer3') as scope:
    W3 = tf.get_variable("W3", shape=[train_width, train_width], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([train_width]), name='bias3')
    layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)

    w3_hist = tf.summary.histogram("weights3", W3)
    b3_hist = tf.summary.histogram("biases3", b3)
    layer3_hist = tf.summary.histogram("layer3", layer3)

with tf.name_scope('layer4') as scope:
    W4 = tf.get_variable("W4", shape=[train_width, train_width], initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([train_width]), name='bias4')
    layer4 = tf.nn.relu(tf.matmul(layer3, W4) + b4)

    w4_hist = tf.summary.histogram("weights4", W4)
    b4_hist = tf.summary.histogram("biases4", b4)
    layer4_hist = tf.summary.histogram("layer4", layer4)

with tf.name_scope('layer5') as scope:
    W5 = tf.get_variable("W5", shape=[train_width, train_width], initializer=tf.contrib.layers.xavier_initializer())
    b5 = tf.Variable(tf.random_normal([train_width]), name='bias5')
    layer5 = tf.nn.relu(tf.matmul(layer4, W5) + b5)

    w5_hist = tf.summary.histogram("weights5", W5)
    b5_hist = tf.summary.histogram("biases5", b5)
    layer5_hist = tf.summary.histogram("layer5", layer5)

with tf.name_scope('hypothesis') as scope:
    W6 = tf.get_variable("W6", shape=[train_width, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
    b6 = tf.Variable(tf.random_normal([nb_classes]), name='bias6')
    hypothesis = tf.sigmoid(tf.matmul(layer5, W6) + b6)

    w6_hist = tf.summary.histogram("weights6", W6)
    b6_hist = tf.summary.histogram("biases6", b6)
    hypothesis_hist = tf.summary.histogram("hypothesis", hypothesis)

with tf.name_scope('cost') as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    cost_summ = tf.summary.scalar('cost', cost)

with tf.name_scope('train') as scope:
    train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
accuracy_summ = tf.summary.scalar('accuracy', accuracy)

with tf.Session() as sess:
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('../logs/mushroom_nn')
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())

    for step in range(1000001):
        summary, _ = sess.run([merged_summary, train], feed_dict={X: x_data, Y: y_data})
        writer.add_summary(summary, global_step=step)

        if step % 100 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_test, Y: y_test})
    print('Hypothesis: ', h)
    print('Correct: ', c)
    print('Accuracy: ', a)
