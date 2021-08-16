import numpy as np
        
def gen_rand_data(data_num = 10000):
    data = (np.random.rand(data_num, 2) - 0.5) * 2
    return data

def get_label(data):
    label = (np.sum(data**2, axis = 1) > 0.71) * 1
    return label

