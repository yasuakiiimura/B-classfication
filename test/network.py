from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.metrics import Precision, Recall

def makemodel():
    model = Sequential()
    model.add(Dense(4, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
    #model.add(Dense(2, activation='relu',kernel_initializer='random_uniform',bias_initializer='zeros'))
    model.add(Dense(1, activation='sigmoid',kernel_initializer='random_uniform',bias_initializer='zeros'))
    model.compile(loss='binary_crossentropy',
                  metrics=[Precision(), Recall(), 'acc'])
    return model


