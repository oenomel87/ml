#   https://www.kaggle.com/uciml/mushroom-classification
#   독버섯 구분 데이터

import tensorflow as tf
import numpy as np

#   정수값으로 계산을 하게 되면 수의 크기가 python에서 다루기 너무 커지므로
#   데이터 값의 크기를 줄임
normalize = 0.1

#   알파벳 데이터를 정수로 변환 하기 위한 수
#   a => 97 이므로 a 일 때 0이 되는 것을 막기 위해 96으로 설정
ascii_diff = 96
edible_code = ord('e') - ascii_diff

raw_data = np.loadtxt('./mushrooms.csv', delimiter=',', dtype=np.character)
data = np.empty((len(raw_data), len(raw_data[0])))

#   알파벳으로 된 데이터를 단순하게 ASCII 코드로 변환
#   'a' => 97, 'b' => 98, 'c' => 99, ...
for i in range(len(raw_data)):
    for j in range(len(raw_data[i])):
        data[i][j] = int(ord(raw_data[i][j]) - ascii_diff)

x_data = np.empty((len(data), len(data[0]) - 1))
y_data = np.empty((len(data), 2))

# y_data = [edible, poison]
for i in range(len(data)):
    #   데이터의 값이 너무 크므로 일정 비율로 줄여준다
    x_data[i] = data[i][1:] * normalize
    
    # is edible?
    if data[i][0] == edible_code:
        y_data[i] = [1, 0]
    else:
        y_data[i] = [0, 1]

#   edible / poison mushroom
nb_classes = 2

X = tf.placeholder("float", [None, len(x_data[0])])
Y = tf.placeholder("float", [None, nb_classes])

W = tf.Variable(tf.random_normal([len(x_data[0]), nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

#   softmax activation
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

#   Cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(100001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 1000 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    # Testing & One-hot encoding
    # edible = 0, poison = 1
    print('---------------------------')
    print('#### poison (1) test')
    poison = sess.run(hypothesis, feed_dict={X: [x_data[0]]})
    print(poison, sess.run(tf.argmax(poison, 1)))
    print('---------------------------')
    print('#### edible (0) test')
    edible = sess.run(hypothesis, feed_dict={X: [x_data[1]]})
    print(edible, sess.run(tf.argmax(edible, 1)))
    print('---------------------------')
    print('#### total test [edible, poison, edible]')
    all = sess.run(hypothesis, feed_dict={X: [x_data[2], x_data[3], x_data[4]]})
    print(all, sess.run(tf.argmax(all, 1)))
