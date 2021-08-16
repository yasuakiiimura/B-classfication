# B-classfication
必要環境:Keras(自分はcolabで実行)

入力データ(x,y)がx*x+y*y>0.5を満たすなら1,満たさないなら0として分類する
datagen.pyで入力データの乱数(0<=x,y<=1)を生成,ラベルをつける
network.py...ネットワーク構造の定義
train.py...学習の実行
