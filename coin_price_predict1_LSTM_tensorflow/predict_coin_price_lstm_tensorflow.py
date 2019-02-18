import tensorflow as tf
import numpy as np
import matplotlib
import os
import time
import pylab
import settings
import data_manager

tf.set_random_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

######## get data ########
#데이터 생성
stock_code = 'XRPBTC'  # coin name
#차트 데이터 준비
chart_data = data_manager.load_chart_data(
    os.path.join(settings.BASE_DIR,
                 './{}.csv'.format(stock_code)))
training_data = data_manager.build_training_data(chart_data)
# 기간 필터링
training_data = training_data[(training_data['date'] >= '09:00 2013-12-27') &
                              (training_data['date'] <= '21:00 2019-01-21')]
training_data = training_data.dropna()
# 차트 데이터 분리
features_chart_data = ['date', 'open', 'high', 'low', 'close', 'volume']
chart_data = training_data[features_chart_data]
# 학습 데이터 분리
features_training_data = [
    'high_low_ratio', 'open_close_ratio',
    'high_open_ratio', 'low_open_ratio',
    'high_close_ratio', 'low_close_ratio',
    'close_lastclose_ratio', 'volume_lastvolume_ratio',
    ]
training_data = training_data[features_training_data]
training_data = training_data[1:]
chart_data = chart_data[1:]

######## set hyper parameter ########
# train Parameters
seq_length = 120
data_dim = 8
hidden_dim = 32
output_dim = 1
learning_rate = 0.0001
iterations = 1000

######## train/test data split ########
#training_data = training_data[0:200] #### test
train_size = int(len(training_data) * 0.5)
train_set = training_data[0:train_size]
test_set = training_data[train_size - seq_length:]  # Index from [train_size - seq_length] to utilize past sequence
train_set.reset_index(drop=True,inplace=True)
test_set.reset_index(drop=True,inplace=True)

######## make input data ########
def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series) - seq_length):
        _x = time_series[i:i + seq_length].values
        _y = time_series['close_lastclose_ratio'][i + seq_length]  # Next close price
        #print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)
    return np.array(dataX), np.array(dataY)

trainX, trainY = build_dataset(train_set, seq_length)
trainY = np.reshape(trainY,(-1,1))
testX, testY = build_dataset(test_set, seq_length)
testY = np.reshape(testY,(-1,1))


######## learn using tensorflow code ########
# input place holders
X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# build a LSTM network
num_layers = 5
def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
stacked_rnn = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)])
outputs, _states = tf.nn.dynamic_rnn(cell=stacked_rnn, inputs=X, dtype=tf.float32)

Y_pred = tf.contrib.layers.fully_connected(
    outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

# cost/loss
loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

# RMSE
targets = tf.placeholder(tf.float32, [None, 1])
predictions = tf.placeholder(tf.float32, [None, 1])
rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

epoch_results, epochs = [], []
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training step
    for i in range(iterations):
        start_time = time.time()
        
        _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
        print("[step: {}] loss: {}".format(i+1, step_loss))
        
        epoch_time = time.time() - start_time
        remain_time = epoch_time * (iterations - (i+1))
        print("epoch_time: %s second" %(round(epoch_time,2))," / remain_time: %s hour" %(round(remain_time/3600,2)))

        if(epoch_time > 1): #한 epoch 당 1초가 넘을때만 plot을 그린다
            epoch_results.append(step_loss)
            epochs.append(i+1)
            pylab.plot(epochs, epoch_results, 'b')
            if not os.path.isdir("./save_graph"):
                os.makedirs("./save_graph")
            pylab.savefig("./save_graph/loss.png")

    # Test step
    test_predict = sess.run(Y_pred, feed_dict={X: testX})
    rmse_val = sess.run(rmse, feed_dict={
                    targets: testY, predictions: test_predict})
    print("RMSE: {}".format(rmse_val))

    # Plot predictions
    plt.plot(testY)
    plt.plot(test_predict)
    plt.xlabel("Time Period")
    plt.ylabel("Stock Price")
    plt.show()

