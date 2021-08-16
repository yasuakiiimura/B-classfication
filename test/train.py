#colab+Google Driveのおまじない
import sys
ROOT_PATH = '/content/drive/My Drive/test'
sys.path.append(ROOT_PATH)

#なかみ
import numpy as np
import matplotlib.pyplot as plt
from datagen import gen_rand_data, get_label
from network import makemodel
from plot import  plotscatter

def result_draw(test_x,test_t,rs):
    plt.cla()
    r_00_x = [x for (x,r,t) in zip(test_x[:,0],rs[:,0],test_t) if (r == t)and(r == 0)]
    r_00_y = [y for (y,r,t) in zip(test_x[:,1],rs[:,0],test_t) if (r == t)and(r == 0)]
    plt.scatter(r_00_x,r_00_y,c='red',marker='o')
    r_01_x = [x for (x,r,t) in zip(test_x[:,0],rs[:,0],test_t) if (r != t)and(r == 0)]
    r_01_y = [y for (y,r,t) in zip(test_x[:,1],rs[:,0],test_t) if (r != t)and(r == 0)]
    plt.scatter(r_01_x,r_01_y,c='red',marker='x')
    r_11_x = [x for (x,r,t) in zip(test_x[:,0],rs[:,0],test_t) if (r == t)and(r == 1)]
    r_11_y = [y for (y,r,t) in zip(test_x[:,1],rs[:,0],test_t) if (r == t)and(r == 1)]
    plt.scatter(r_11_x,r_11_y,c='blue',marker='o')
    r_10_x = [x for (x,r,t) in zip(test_x[:,0],rs[:,0],test_t) if (r != t)and(r == 1)]
    r_10_y = [y for (y,r,t) in zip(test_x[:,1],rs[:,0],test_t) if (r != t)and(r == 1)]
    plt.scatter(r_10_x,r_10_y,c='blue',marker='x')
    plt.savefig("result.png", format="png", dpi=300)
    return (len(r_00_x)+len(r_11_x))/len(test_x)

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
