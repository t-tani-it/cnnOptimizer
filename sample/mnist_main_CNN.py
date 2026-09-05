import numpy as np
import time

from tinydb import TinyDB

from sklearn.model_selection import train_test_split

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers import BatchNormalization
from keras.optimizers import adam_v2
from keras.utils import np_utils
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import backend as K

################################################################
import tensorflow as tf
# from keras.utils.all_utils import get_custom_objects
from keras.utils.generic_utils import get_custom_objects

class Mish(Activation):
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))

class tanhExp(Activation):
    def __init__(self, activation, **kwargs):
        super(tanhExp, self).__init__(activation, **kwargs)
        self.__name__ = 'tanhExp'
def tanhexp(x):
    return x * tf.math.tanh(tf.math.exp(x))

get_custom_objects().update({'tanhExp':tanhExp(tanhexp)})
get_custom_objects().update({'Mish':Mish(mish)})
################################################################

# def Swish(x, beta=1.):
#     return x * K.sigmoid(beta * x)
# def tanhExp(x):
#     return x * Activation.tanh(np.exp(x))
# def Mish(x):
#     return x * Activation.tanh(Activation.softplus(x))
# def x_(x):
#     return x

# def activa(act):
#     # print('activa --> ' + act)
#     # input()
#     if act == 'gelu':
#         return 'gelu'

def opt(mome):
    # オプティマイズ、モーメントを構築
    if mome == 'adam':
        return adam_v2.Adam(learning_rate=0.001,
                            beta_1 = 0.9, beta_2 = 0.999)
    elif mome == 'amsgrad':
        # print(mome)
        # input()
        return adam_v2.Adam(learning_rate=0.001,
                            beta_1 = 0.9, beta_2 = 0.999, amsgrad = True)

def fit_loop(dom1,dom2,dom3,k_size1,k_size2,do,act,mome,loss,epoch,batch_s,rota_rng,w_sht_rng,h_sht_rng,l_zoom_rng,filename,start_time):

    # input()
    # 記録用のリストを定義
    # loss_list = []
    # acc_list = []
    time_processed = []

    # テストデータをメモ帳へアウトプットするための準備
    s = ""
    s = s + '\ndom1,dom2,dom3,k_size1,k_size2,do,act,mome,loss,epoch,batch_s,rota_rng,w_sht_rng,h_sht_rng,l_zoom_rng,filename,start_time,'

    # MNISTのデータを読み込む --- (※1)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # print('X_tr' + str(np.shape(X_train)))
    # print('y_tr' + str(np.shape(y_train)))

    # データをfloat32型に変換して正規化する --- (※2)
    X_train = X_train.reshape(60000, 28, 28, 1).astype('float32')
    X_test  = X_test.reshape(10000, 28, 28, 1).astype('float32')
    X_train /= 255
    X_test  /= 255
    # ラベルデータを0-9までのカテゴリを表す配列に変換 --- (*2a)
    y_train = np_utils.to_categorical(y_train, 10)
    y_test  = np_utils.to_categorical(y_test, 10)

    # ////////////////////////////////////////////////////////////////////////////
    # テストデータを分ける
    random_seed = 2
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = random_seed)
    # ////////////////////////////////////////////////////////////////////////////

    # 時間計測開始
    time_start = time.time()

    # モデルの構造を定義 --- (※3)
    model = Sequential()

    model.add(Conv2D(dom1, kernel_size = k_size1, padding='same', input_shape=(28, 28, 1)))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(do))

    model.add(Conv2D(dom2, kernel_size = k_size2, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation(act))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(do))

    model.add(Conv2D(dom2, kernel_size = k_size2, padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation(act))
    # model.add(Dropout(0.25))                                    # 効果があるかなんとも言えない…

    model.add(Flatten())
    model.add(Dense(dom3, activation=act))
    # model.add(Dropout(0.25))                                    # 効果があるかなんとも言えない…

    model.add(Dense(10))
    model.add(Activation('softmax'))

    # model.summary()


    model.compile(
        loss=loss,
        optimizer = opt(mome),
        metrics=['accuracy'])

    verbose_ = 2
    # データで訓練 --- (※5)
    es = EarlyStopping(monitor = 'val_loss',
                        patience = 5,
                        verbose = verbose_)
    # ReduceLROnPlateau の設定
    lr = ReduceLROnPlateau(
                            monitor = 'val_loss',
                            factor = 0.2,
                            patience = 4,
                            min_lr = 0.0001,
                            cooldown = 2,
                            verbose = verbose_
                            )
    # ImageDataGenerator 設定
    datagen = ImageDataGenerator(
        featurewise_center=False,               # データセット全体で,入力の平均を0に調整
        samplewise_center=False,                # 各サンプルの平均を0に調整
        featurewise_std_normalization=False,    # 入力をデータセットの標準偏差で正規化
        samplewise_std_normalization=False,     # 各入力をその標準偏差で正規化
        zca_whitening=False,                    # ZCA白色化のイプシロン
        rotation_range=rota_rng,                # 回転角度(-30～30度)
        width_shift_range=w_sht_rng,            # 左右のスライド幅
        height_shift_range=h_sht_rng,           # 上下のスライド幅
        zoom_range=l_zoom_rng,                  # 拡大・縮小率
        horizontal_flip=False,                  # 水平反転しない
        vertical_flip=False)                    # 垂直反転しない
    # 水増し画像を訓練用画像の形式に合わせる
    datagen.fit(X_train)

    # hist = model.fit(X_train, y_train,
    #                     epochs = 5, batch_size = 128,
    #                     verbose=verbose_,
    #                 #  validation_data = (x_val, t_val)
    #                     callbacks = [es])

    # 水増し画像訓練データで訓練 --- (※5)
    hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_s),
                                            epochs = epoch,
                                            verbose = verbose_,
                                            validation_data = (X_val, y_val),
                                            callbacks = [es, lr])
    # テストデータを用いて評価する --- (※6)
    score = model.evaluate(X_test, y_test, verbose=verbose_)
    print('loss=', score[0])
    print('accuracy=', score[1])

# モデルを保存
    model.save_weights('MNIST_CNN_{0:.4f}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}_{14}.hdf5'.format(score[1],dom1,dom2,dom3,k_size1,k_size2,do,mome,act,epoch,batch_s,rota_rng,w_sht_rng,h_sht_rng,l_zoom_rng))
    # model.save_weights('MNIST_CNN_{0:.4f}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}.hdf5'.format(score[1],dom1,dom2,do,mome,act,epoch,batch_s,rota_rng,w_sht_rng,h_sht_rng,l_zoom_rng))


    #時間計測終了、記録用リストへの各種データの追加
    time_stop = time.time()
    time_processed.append(time_stop - time_start)

    # テストデータをメモ帳へアウトプット
    # s = ""                    4loop外に持って行った（↑）
    # s = s + '\ndom,do,layerist,mome,act,loss,loss_score,accuracy_score,time'
    s = s + ','.join([str(dom1),str(dom2),str(dom3),str(do),str(mome),str(act),str(loss),str(epoch),str(batch_s), \
                str(rota_rng),str(w_sht_rng),str(h_sht_rng),str(l_zoom_rng),\
                    str(score[0]),str(score[1]),str(time_processed[0]),start_time])
    # s = s +',{0:.4f}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}_{14}_{15}.hdf5'.format(score[1],dom1,dom2,dom3,k_size1,k_size2,do,mome,act,epoch,batch_s,rota_rng,w_sht_rng,h_sht_rng,l_zoom_rng)

    with open(filename, mode = 'a') as f:
        # print(s)
        # input()
        f.write(s)
        print('-----書き込み完了')

    # input()
    filepath = "tinydb.json"  # 旧絶対パスを可搬化(当時の記録のため動作保証外)
    db = TinyDB(filepath)

    # 既存のデータがあれば破棄
    # db.drop_table('CNN')

    # テーブルを得る
    table = db.table('CNN')

    # データをデータベースに挿入
    table.insert({'dom1':dom1,'dom2':dom2,'dom3':dom3,'k_size1':k_size1,'k_size2':k_size2,\
                'do':do,'act':act,'mome':mome,'loss':loss,'epoch':epoch,'batch_s':batch_s,\
                'rota_rng':rota_rng,'w_sht_rng':w_sht_rng,'h_sht_rng':h_sht_rng,'l_zoom_rng':l_zoom_rng,\
                'filename':filename,'start_time':start_time,\
                'loss':score[0],'act':score[1]})

    # 全データを抽出して表示
    # print(table.all())
