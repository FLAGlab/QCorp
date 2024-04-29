avgReward = 0
while (avgReward / 100) < 195:
    import gym

    env = gym.make('CartPole-v1')
    import numpy as np

    import tensorflow as tf

    y = tf.placeholder(tf.float32, shape=[None, 4])

    m1 = tf.Variable(tf.truncated_normal(shape=[4, 16]))
    m2 = tf.Variable(tf.truncated_normal(shape=[16, 8]))
    m3 = tf.Variable(tf.truncated_normal(shape=[8, 4]))

    x = tf.placeholder(tf.float32, shape=[None, 4])

    mx_b_ie_Qtable = tf.matmul(x, m1)
    mx_b_ie_Qtable = tf.matmul(mx_b_ie_Qtable, m2)
    mx_b_ie_Qtable = tf.matmul(mx_b_ie_Qtable, m3)

    loss = tf.reduce_mean(tf.square(mx_b_ie_Qtable - y))
    trainingStep = tf.train.AdamOptimizer(0.01).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    state = 0
    failed = False
    wlf = 0
    for i in range(2700):
        feature = env.reset()
        while not failed:
            import random

            featureNP = np.reshape(np.array(feature), [-1, 4])
            # state = random.choice(range(2))
            action = random.choice(range(2))
            featureForNext, reward, failed, _ = env.step(action)
            Qtabel = sess.run(mx_b_ie_Qtable, feed_dict={x: featureNP})
            Qtabel = np.reshape(np.array(Qtabel), [2, 2])
            Qtabel[state, action] = reward + 0.97 * np.max(Qtabel[action])
            labelNP = np.reshape(np.array(Qtabel), [-1, 4])
            for j in range(10):
                sess.run([trainingStep, loss],
                         feed_dict={x: featureNP, y: labelNP})
                wlf = sess.run(loss,
                               feed_dict={x: featureNP, y: labelNP})
            # env.render()
            feature = featureForNext
            state = action
    highestReward = 0
    avgReward = 0
    totalReward = 0
    for i in range(100):
        highestReward = totalReward if totalReward > highestReward else highestReward
        avgReward = avgReward + totalReward
        totalReward = 0
        feature = env.reset()
        failed = False
        state = 0
        while not failed:
            featureNP = np.reshape(np.array(feature), [-1, 4])
            Qtabel = sess.run(mx_b_ie_Qtable, feed_dict={x: featureNP})
            Qtabel = np.reshape(np.array(Qtabel), [2, 2])
            action = np.argmax(Qtabel[state])
            # print(action, "**")
            featureForNext, reward, failed, _ = env.step(action)
            feature = featureForNext
            # env.render()
            totalReward = reward + totalReward
            # print(totalReward)
            state = action

    print(highestReward, avgReward / 100, wlf)