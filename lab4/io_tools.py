import numpy as np

def readFile(file):
    with open('data/' + file) as f:
        raw_images = f.readlines()
        images = []
        for raw_img in raw_images:
            img = np.array([int(i.strip()) for i in raw_img.split(',')])
            images.append(img)
        images = np.array(images)
        return images

'''
    8000 vectors with 784 1/0 in each.
    One image: 784 pixels = 28x28
    returns test, train
            np.array(8000, 784), np.array(2000, 784)
'''
def get_training_data():
    train_X = readFile('bindigit_trn.csv')
    train_Y = readFile('targetdigit_trn.csv')
    return train_X, train_Y

'''
    8000 vectors with 784 1/0 in each.
    One image: 784 pixels = 28x28
    returns test, train
            np.array(8000, 784), np.array(2000, 784)
'''
def get_testing_data():
    test_X = readFile('bindigit_tst.csv')
    test_Y =  readFile('targetdigit_tst.csv')
    return test_X, test_Y
