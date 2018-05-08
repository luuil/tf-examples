import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

learning_rate = 0.001
training_steps = 500
display_steps = 50

# training data
X_train = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,
    2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1])
Y_train = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,
    2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = X_train.shape[0]

# tf input
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# model
# weight
W = tf.Variable(np.random.randn(), name='weight')
b = tf.Variable(np.random.randn(), name='bias')

# linear model
y = tf.add(tf.multiply(X, W), b)

# minimize squares error
cost = tf.reduce_mean(tf.square(y-Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

# init variables
init = tf.initialize_all_variables()

# launch the graph
with tf.Session() as sess:
    sess.run(init)

    # fit all train data
    for epoch in range(training_steps):
        for (x, y) in zip(X_train, Y_train):
            sess.run(train, feed_dict={X: x, Y: y})

        # display every 50 steps
        if epoch%display_steps == 0:
            print('Epoch: %04d' % (epoch+1), \
                'cost = {:.9f}'.format(sess.run(cost, feed_dict={X: X_train, Y: Y_train})), \
                'W = {:.9f}'.format(sess.run(W)), \
                'b = {:.9f}'.format(sess.run(b)))

    print('Optimize finished!')
    print('cost = {:.9f}'.format(sess.run(cost, feed_dict={X: X_train, Y: Y_train})), \
        'W = {:.9f}'.format(sess.run(W)), \
        'b = {:.9f}'.format(sess.run(b)))

    # graphic display
    plt.plot(X_train, Y_train, 'ro', label='Original data')
    plt.plot(X_train, sess.run(W)*X_train+sess.run(b), label='Fitted data')
    plt.legend()
    plt.show()