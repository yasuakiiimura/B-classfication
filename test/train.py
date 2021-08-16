#colab+Google Driveのおまじない
import sys
ROOT_PATH = '/content/drive/My Drive/test'
sys.path.append(ROOT_PATH)

#なかみ
import numpy as np
import matplotlib.pyplot as plt
from datagen import gen_rand_data, get_label
from network import makemodel
from plot import  plotscatter,result_draw

Model = makemodel()
train_x = gen_rand_data(100000)
train_t = get_label(train_x)
test_x = gen_rand_data(1000)
test_t = get_label(test_x)

plotscatter(train_x,train_t)

history = Model.fit(train_x, train_t, epochs=100,batch_size=100)

plt.cla()
plt.plot(history.epoch, history.history["acc"], label="acc")
plt.xlabel("epoch")
plt.savefig("acc.png", format="png", dpi=300)
plt.cla()
plt.plot(history.epoch, history.history["loss"], label="loss")
plt.xlabel("epoch")
plt.savefig("loss.png", format="png", dpi=300)
plt.legend()

rs = Model.predict(test_x, batch_size=None, verbose=0, steps=None)
rs = (rs > 0.5)*1

#t==yのやつとそうでないやつを01ごとに分ける->4分類する
acc = result_draw(test_x,test_t,rs)

print("Test accuracy", acc)
