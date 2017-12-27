#   https://www.kaggle.com/uciml/mushroom-classification
#   독버섯 구분 데이터

import tensorflow as tf
import numpy as np

raw_data = np.loadtxt('./mushrooms.csv', delimiter=',', dtype=np.character)
data = np.empty((len(raw_data), len(raw_data[0])))

#   알파벳으로 된 데이터를 단순하게 ASCII 코드로 변환
#   'a' => 97, 'b' => 98, 'c' => 99, ...
for i in range(len(raw_data)):
    for j in range(len(raw_data[i])):
        data[i][j] = int(ord(raw_data[i][j]) - 97)
'''
    data =
        [[ 15.  23.  18. ...,  10.  18.  20.]
        [  4.  23.  18. ...,  13.  13.   6.]
        [  4.   1.  18. ...,  13.  13.  12.]
        ..., 
        [  4.   5.  18. ...,   1.   2.  11.]
        [ 15.  10.  24. ...,  22.  21.  11.]
        [  4.  23.  18. ...,  14.   2.  11.]]
'''

x_data = np.empty((len(data), len(data[0]) - 1))
y_data = np.empty((len(data), 2))

# y_data = [edible, poison]
for i in range(len(data)):
    x_data[i] = data[i][1:]
    # is edible?
    if data[i][0] == 4:
        y_data[i] = [1, 0]
    else:
        y_data[i] = [0, 1]
'''
    x_data =
        [[ 15.  23.  18. ...,  10.  18.  20.]
        [  4.  23.  18. ...,  13.  13.   6.]
        [  4.   1.  18. ...,  13.  13.  12.]
        ..., 
        [  4.   5.  18. ...,   1.   2.  11.]
        [ 15.  10.  24. ...,  22.  21.  11.]
        [  4.  23.  18. ...,  14.   2.  11.]]

    y_data =
        [[ 0.  1.]
        [ 1.  0.]
        [ 1.  0.]
        ..., 
        [ 1.  0.]
        [ 0.  1.]
        [ 1.  0.]]
'''


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
    
    for step in range(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

        '''
        (0, nan)
        (200, nan)
        (400, nan)
        (600, nan)
        .
        .
        .
        '''
