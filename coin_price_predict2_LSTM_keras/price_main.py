import os
import settings
import data_manager
import numpy as np
from price_network import PriceNetwork
from keras.models import load_model
import matplotlib.pyplot as plt
import pylab

if __name__ == '__main__':
    #매번 조정해줄 변수들
    stock_code = 'TUSD'  # coin name
    model_ver = '' # model to load


    #model save/load 에사용할 정보
    model_code = 'price'
    price_model_ver = 'price_{}'.format(model_ver)
    timestr = settings.get_time_str()

########################### importing data part ###########################
    # 차트 데이터 준비
    chart_data = data_manager.load_chart_data('./{}BTC.csv'.format(stock_code))
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

########################### manipulating data part ###########################

    ######## ★set hyper parameter ########
    seq_length = 120
    data_dim = 8
    batch_size = 1000
    
    learning_rate = 0.0001
    num_epochs = 1000

    ######## train/test split ########
    train_set = training_data[0:150] #### test
    test_set = training_data[30:300] ### test
    test_set.reset_index(drop=True,inplace=True) ### test
    '''
    train_size = int(len(training_data) * 0.6)
    train_set = training_data[0:train_size]
    valid_size = int(len(training_data) * 0.2)
    valid_set = training_data[train_size-seq_length : train_size + valid_size]
    test_set = training_data[(train_size + valid_size) - seq_length:]  # Index from [train_size - seq_length] to utilize past sequence
    train_set.reset_index(drop=True,inplace=True)
    valid_set.reset_index(drop=True,inplace=True)
    test_set.reset_index(drop=True,inplace=True)
    '''

    ######## make input data ########
    def build_dataset(time_series, seq_length):
        dataX = []
        dataY = []
        for i in range(0, len(time_series) - seq_length - 1):
            _x = time_series[i:i + seq_length].values
            _y = time_series['close_lastclose_ratio'][i + seq_length + 1]  # Next close price
            #print(_x, "->", _y)
            dataX.append(_x)
            dataY.append(_y)
        return np.array(dataX), np.array(dataY)

    trainX, trainY = build_dataset(train_set, seq_length)
    trainY = np.reshape(trainY,(-1,1))
    testX = trainX ## test
    testY = trainY ## test
    #validX, validY = build_dataset(valid_set, seq_length)
    #validY = np.reshape(validY,(-1,1))
    #testX, testY = build_dataset(test_set, seq_length)
    #testY = np.reshape(testY,(-1,1))

########################### training part ###########################
    # 학습 시작
    price_network = PriceNetwork(seq_length=seq_length, data_dim=data_dim, output_dim=1, lr=learning_rate)

    # 여기가 핵심
    #hist = price_network.model.fit(trainX, trainY, epochs=num_epochs, batch_size=batch_size, validation_data=(validX,validY), verbose=2)
    hist = price_network.model.fit(trainX, trainY, epochs=num_epochs, batch_size=batch_size, verbose=2)
    '''
    # 가격 예측 신경망을 파일로 저장
    model_dir = os.path.join(settings.BASE_DIR, 'models/%s' % model_code)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, 'model_price_%s.h5' % timestr)
    price_network.model.save(model_path)
    '''

    '''
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','val'],loc='upper left')
    plt.show()
    '''
    #pylab.plot(hist.history['loss'])
    #if not os.path.isdir("./save_graph"):
    #    os.makedirs("./save_graph")
    #pylab.savefig("./save_graph/loss.png")

    test_predict = price_network.model.predict(testX)
    plt.plot(np.concatenate(testY), color='red')
    plt.plot(np.concatenate(test_predict), color='blue')
    plt.xlabel("Time Period")
    plt.ylabel("Coin Price")
    plt.show()

    

