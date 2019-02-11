import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.utils import shuffle
import datetime
import time
import re

train = pd.read_csv('./dataSource/jinnan_round1_train_20181227.csv', header=0, encoding='utf-8')
test = pd.read_csv('./dataSource/jinnan_round1_testA_20181227.csv', header=0, encoding='utf-8')
test2 = pd.read_csv('./dataSource/jinnan_round1_testB_20190121.csv', header=0, encoding='utf-8')

def get_weights(shape, lambd):
    var = tf.Variable(tf.zeros(shape), dtype=tf.float32, name='variable_weight')
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambd)(var))
    return var

def add_layer(input, in_size, out_size, activation_function=None, dropout_function=False):
    Weights = get_weights([in_size, out_size], 0.1)
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.01, name='variable_biases')
    Wx_plus_b = tf.matmul(input, Weights) + biases

    if dropout_function == True:
        Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob=0.5)
    else:
        pass

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

#提取时间特征
def data_feature(df,data1,data2):
    begin = ''
    end = ''
    if df[data1] !=df[data1] or df[data2] !=df[data2]:
        return np.NaN
    matchObj1 = re.match( r'\d{2}:\d{2}:\d{2}', df[data1], re.M|re.I)
    matchObj1_0 = re.match( r'\d{1}:\d{2}:\d{2}', df[data1], re.M|re.I)
    matchObj2 = re.match( r'\d{2}:\d{2}:\d{2}', df[data2], re.M|re.I)
    matchObj2_0 = re.match( r'\d{1}:\d{2}:\d{2}', df[data2], re.M|re.I)

    if '-' in str(df[data1]):
        t = df[data1].split('-')[1]
        matchObj1_1 = re.match( r'\d{2}:\d{2}', t, re.M|re.I)
        matchObj1_1_0 = re.match( r'\d{1}:\d{2}', t, re.M|re.I)
        if matchObj1_1:
            begin = re.findall(r'\d{2}:\d{2}', t)[0] + ':00'
        elif matchObj1_1_0:
            begin = re.findall(r'\d{1}:\d{2}', t)[0] + ':00'
        else:
            return np.NaN
    elif matchObj1:
        begin = re.findall(r'\d{2}:\d{2}:\d{2}',df[data1])[0]
    elif matchObj1_0:
        begin = re.findall(r'\d{1}:\d{2}:\d{2}',df[data1])[0]
    else:
        return np.NaN

    if '-' in str(df[data2]):
        t = df[data2].split('-')[0]
        matchObj1_2 = re.match( r'\d{2}:\d{2}', t, re.M|re.I)
        matchObj1_2_0 = re.match( r'\d{1}:\d{2}', t, re.M|re.I)
        if matchObj1_2:
            end = re.findall(r'\d{2}:\d{2}', t)[0]  + ':00'
        elif matchObj1_2_0:
            end = re.findall(r'\d{1}:\d{2}', t)[0]  + ':00'
        else:
            return np.NaN
    elif matchObj2:
        end = re.findall(r'\d{2}:\d{2}:\d{2}',df[data2])[0]
    elif matchObj2_0:
        end = re.findall(r'\d{1}:\d{2}:\d{2}',df[data2])[0]
    else:
        return np.NaN

    if begin == '24:00:00':
        begin = '23:59:59'
    if end == '24:00:00':
        end = '23:59:59'
    begin = datetime.datetime.strptime(begin, '%H:%M:%S')
    end = datetime.datetime.strptime(end, '%H:%M:%S')
    result = round((end - begin).seconds/60)
    return result

def is_numeric(data):
    if data.isdigit():
        return True
    else:
        return False

def getData(df):
    columns = ['A1','A3','A4','A6','A10','A12','A13','A15','A17','A18','A19',
    'A21','A22','A23','A25','A27','B1','B2','B3','B6','B8','B12','B13','B14',
    'A9-A5','A11-A9','A14-A11','A16-A14','A20-A16','A24-A20','A26-A24','A28-A26','B5-B4','B7-B5','B9-B7','B10-B9']
    label=['rate']
    df = df[~df['A25'].str.contains('1900')]
    sum = df.shape[0]
    df = shuffle(df)
    train_x = pd.DataFrame(df, columns=columns)
    pd.DataFrame(train_x, columns=columns).to_csv('train_x.csv', index=False)
    train_x = np.array(train_x)
    # minmax = preprocessing.MinMaxScaler()
    # train_x = minmax.fit_transform(train_x)
    # train_x = preprocessing.scale(train_x)
    pd.DataFrame(train_x, columns=columns).to_csv('scale_train_x.csv', index=False)
    train_y = np.array(df[label])

    x = tf.placeholder('float', [None, 36])
    y_ = tf.placeholder('float', [None, 1])
    y = add_layer(x, 36, 36)
    y = add_layer(x, 36, 10, activation_function=tf.nn.relu)
    y = add_layer(y, 10, 1, activation_function=tf.nn.sigmoid)
    
    cross_entropy = tf.reduce_mean(tf.square(y_-y))
    tf.add_to_collection("losses", cross_entropy)
    loss = tf.add_n(tf.get_collection("losses"))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        m_saver = tf.train.Saver()
        for i in range(0, sum, 10):
            batch_X = train_x[i:i + 10]
            batch_y = train_y[i:i + 10]
            if i % 50 == 0:
                #print('train percent :' + str(round(i*100/sum, 2)) + '%')
                print('loss is :' + str(sess.run(loss, feed_dict={x: batch_X, y_: batch_y})))
                m_saver.save(sess, './dataSource/model', global_step=i)
            sess.run(train_step, feed_dict={x: batch_X, y_: batch_y})

def getPredict(df):
    tf.reset_default_graph()
    columns = ['A1','A3','A4','A6','A10','A12','A13','A15','A17','A18','A19',
    'A21','A22','A23','A25','A27','B1','B2','B3','B6','B8','B12','B13','B14',
    'A9-A5','A11-A9','A14-A11','A16-A14','A20-A16','A24-A20','A26-A24','A28-A26','B5-B4','B7-B5','B9-B7','B10-B9']
    df['A25'].fillna(df['A25'].median(), inplace=True)
    df['A27'].fillna(df['A27'].median(), inplace=True)
    result = pd.DataFrame(df,columns=['id'])
    train_x = pd.DataFrame(df, columns=columns)
    train_x = np.array(train_x)
    # minmax = preprocessing.MinMaxScaler()
    # train_x = minmax.fit_transform(train_x)
    # train_x = preprocessing.scale(train_x)
    x = tf.placeholder('float', [None, 36])
    y = add_layer(x, 36, 36)
    y = add_layer(x, 36, 10, activation_function=tf.nn.relu)
    y = add_layer(y, 10, 1, activation_function=tf.nn.sigmoid)
    tmp=np.array([])
    with tf.Session() as sess:
        my_saver = tf.train.Saver()
        my_saver.restore(sess, './dataSource/model-1350')
        t = sess.run(y, feed_dict={x: train_x}).reshape(1,-1)[0]
        tmp = np.concatenate((tmp, t), axis=0)
        result['rate'] = np.array(tmp)
        result.to_csv(str(datetime.datetime.now().second) + 'test.csv', index=False)

def do(train,bo):

    #删除缺失值过多的列
    train.drop(['A2','A7','A8','B11'], axis=1, inplace=True)

    #处理数字型缺失值
    #train['B11'] = 60
    train['A3'].fillna(train['A3'].median(), inplace=True)
    train['A21'].fillna(train['A21'].median(), inplace=True)
    train['A23'].fillna(train['A23'].median(), inplace=True)
    train['B1'].fillna(train['B1'].median(), inplace=True)
    train['B2'].fillna(train['B2'].median(), inplace=True)
    train['B3'].fillna(train['B3'].median(), inplace=True)
    train['B8'].fillna(train['B8'].median(), inplace=True)
    train['B12'].fillna(train['B12'].median(), inplace=True)
    train['B13'].fillna(train['B13'].median(), inplace=True)

    #特征提取
    train['A9-A5'] = train.apply(data_feature, axis = 1, args=('A5','A9'))
    train['A11-A9'] = train.apply(data_feature, axis = 1, data1='A9',data2='A11')
    train['A14-A11'] = train.apply(data_feature, axis = 1, data1='A11',data2='A14')
    train['A16-A14'] = train.apply(data_feature, axis = 1, data1='A14',data2='A16')
    train['A20-A16'] = train.apply(data_feature, axis = 1, data1='A16',data2='A20')
    train['A24-A20'] = train.apply(data_feature, axis = 1,data1='A20',data2='A24')
    train['A26-A24'] = train.apply(data_feature, axis = 1, data1='A24',data2='A26')
    train['A28-A26'] = train.apply(data_feature, axis = 1, data1='A26',data2='A28')
    train['B5-B4'] = train.apply(data_feature, axis = 1, data1='B4',data2='B5')
    train['B7-B5'] = train.apply(data_feature, axis = 1, data1='B5',data2='B7')
    train['B9-B7'] = train.apply(data_feature, axis = 1, data1='B7',data2='B9')
    train['B10-B9'] = train.apply(data_feature, axis = 1, data1='B9',data2='B10')

    #处理特征提取后的缺失值
    train['A9-A5'].fillna(train['A9-A5'].median(), inplace=True)
    train['A11-A9'].fillna(train['A11-A9'].median(), inplace=True)
    train['A14-A11'].fillna(train['A14-A11'].median(), inplace=True)
    train['A16-A14'].fillna(train['A16-A14'].median(), inplace=True)
    train['A20-A16'].fillna(train['A20-A16'].median(), inplace=True)
    train['A24-A20'].fillna(train['A24-A20'].median(), inplace=True)
    train['A26-A24'].fillna(train['A26-A24'].median(), inplace=True)
    train['A28-A26'].fillna(train['A28-A26'].median(), inplace=True)
    train['B5-B4'].fillna(train['B5-B4'].median(), inplace=True)
    train['B7-B5'].fillna(train['B7-B5'].median(), inplace=True)
    train['B9-B7'].fillna(train['B9-B7'].median(), inplace=True)
    train['B10-B9'].fillna(train['B10-B9'].median(), inplace=True)
    if bo:
        getData(train)
    else:
        getPredict(train)

def init():
    do(train, True)
    do(test, False)
    do(test2, False)

init()