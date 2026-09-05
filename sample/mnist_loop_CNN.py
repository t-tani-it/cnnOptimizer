import mnist_main_CNN     #別モジュール（loopさせるやつ）をインポート
import time
import datetime

# dom1_d = {'dom':[64]}
# dom2_d = {'dom':[200,300,400]}
# # dom2_d = {'dom':[500,600,700]}
# # dom2_d = {'dom':[128,246,512,1000,1024]}
# do_d = {'do':[0.25,0.3]}
# # do_d = {'do':[0.35,0.4,0.45,0.5]}
# layer_l = [0]                                       #NN用
# # mome_d = {'mome':['amsgrad','adam']}
# mome_d = {'mome':['amsgrad']}
# # act_d = {'act':['gelu','swish','tanhExp','Mish','relu']}
# act_d = {'act':['gelu']}
# loss_d = {'loss':['categorical_crossentropy']}
# epochs = [40]
# batch_sizes = [128]
# rota_rngs = [20]
# w_sht_rngs = [0.3]
# h_sht_rngs = [0.3]
# l_zoom_rng = [0.9,1.5]
# filename = 'keras_loss_acc_CNN_64_b128_40_kai4.txt'


dom1_d = {'dom':[64]}
dom2_d = {'dom':[400,500,600,700]}
# dom2_d = {'dom':[100,200,300,400,500,600,700]}
# dom1_d = {'dom':[32,64]}
# dom2_d = {'dom':[128,246,512,1024]}
do_d = {'do':[0.25,0.3]}
# do_d = {'do':[0.35,0.4,0.45,0.5]}
layer_l = [0]                                       #NN用
mome_d = {'mome':['amsgrad']}
# act_d = {'act':['gelu','relu','swish']}
act_d = {'act':['relu']}
loss_d = {'loss':['categorical_crossentropy']}
epochs = [40]
batch_sizes = [86]
rota_rngs = [10]
w_sht_rngs = [0.1]
h_sht_rngs = [0.1]
# l_zoom_rng = [0.9, 1.5]
l_zoom_rng = 0.1

lr_ONOFF = True
# lr_ONOFF = False
sk_split = True

filename = 'keras_loss_acc_CNN_d{0}_b{1}_e{2}_kai5(imggen_{3}_{4})_validation_true.txt'.format(dom1_d['dom'],batch_sizes,epochs,lr_ONOFF,sk_split)
# filename = 'keras_loss_acc_CNN_64_b128_40_kai3.txt'


# # dom_d = {'dom':[100,200,300,400,500,600,700]}
# dom1_d = {'dom':[16]}
# dom2_d = {'dom':[100,128,200,256]}
# do_d = {'do':[0.1,0.2,0.3,0.4,0.5]}
# # do_d = {'do':[0.35,0.4,0.45,0.5]}
# layer_l = [0]                                       #NN用
# mome_d = {'mome':['adam']}
# # act_d = {'act':['gelu','relu','swish']}
# act_d = {'act':['relu']}
# loss_d = {'loss':['categorical_crossentropy']}
# epochs = [2]
# # batch_sizes = [128]
# batch_sizes = [1280]
# # -----imggenに関する設定
# rota_rngs = [10]
# w_sht_rngs = [0.1]
# h_sht_rngs = [0.1]
# # l_zoom_rng = [0.9, 1.5]
# l_zoom_rng = 0.1
# # -----保存するファイルの名前
# filename = 'keras_loss_acc_CNN_tmp.txt'

# 時間関係の記録
time_str = time.time()
time_str_str = str(time_str)

dt_stt = datetime.datetime.now()
dt_stt.strftime('%Y/%m/%d %H:%M:%S')
dt_stt_str = str(dt_stt)

# スタートの記録をメモ帳へ
s = '\ntime_str_str --> ' + time_str_str
s = s + '\nstart --> ' + dt_stt_str
with open(filename, mode = 'a') as f:
    # print(s)
    # input()
    f.write(s)
    print('-----書き込み完了')

# ステータスバーの準備
bunbo = len(dom1_d['dom']) * len(dom2_d['dom']) * len(do_d['do']) * len(layer_l) * \
        len(mome_d['mome']) * len(act_d['act']) * len(loss_d['loss']) * len(epochs) * len(batch_sizes)
bunsi = 1

for dom1 in dom1_d['dom']:
    # input()
    for dom2 in dom2_d['dom']:
        for do in do_d['do']:
            for layer in layer_l:
                # input()
                for mome in mome_d['mome']:
                    for act in act_d['act']:
                        for loss in loss_d['loss']:
                            for epoch in epochs:
                            # mnist_main.fit_loop(dom,do,layer_d['layer'],mome,act,loss)
                                for batch_size in batch_sizes:
                                    for rota_rng in rota_rngs:
                                            for w_sht_rng in w_sht_rngs:
                                                    for h_sht_rng in h_sht_rngs:
                                                            mnist_main_CNN.fit_loop(dom1,dom2,do,layer,mome,act,loss,epoch,batch_size,\
                                                                                    rota_rng,w_sht_rng,h_sht_rng,l_zoom_rng,\
                                                                                    filename,dt_stt_str)
                                                            # 時間関係の記録
                                                            dt_stop = datetime.datetime.now()
                                                            dt_pro = dt_stop - dt_stt
                                                            prediction_end_time = dt_stt + dt_pro * bunbo/bunsi
                                                            print(f'status bar -->  {bunsi} / {bunbo} 完了')
                                                            print(f'prediction_end_time -->  {prediction_end_time} 完了予定')
                                                            print(f'elapsed_time -->  {dt_pro} 時間経過')
                                                            # print(f'end_time -->  {end_time} 完了')
                                                            bunsi += 1

# # 時間関係の記録
time_stop = time.time()
time_pro = time_stop - time_str
time_stop = str(time_stop)
time_pro = str(time_pro)

dt_stp = datetime.datetime.now()
dt_stp.strftime('%Y/%m/%d %H:%M:%S')
dt_stp = str(dt_stp)
print('time_start --> ' + time_str_str)
print('time_stop --> ' + time_stop)
print('datetime_start --> ' + dt_stt_str)
print('datetime_stop --> ' + dt_stp)
print('time_pro --> ' + time_pro)

#記録をメモ帳へ
s = '\ntime_pro --> ' + time_pro
s = s + '\nstart --> ' + dt_stt_str
s = s + '\nstop --> ' + dt_stp + '\n'
with open(filename, mode = 'a') as f:
    # print(s)
    # input()
    f.write(s)
    print('-----書き込み完了')