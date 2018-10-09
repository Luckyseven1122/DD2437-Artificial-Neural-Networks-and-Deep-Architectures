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
def getBindigit():
    bindigit_trn = readFile('bindigit_trn.csv')
    bindigit_tst = readFile('bindigit_tst.csv')
    return bindigit_trn, bindigit_tst

''' 
    8000 vectors with 784 1/0 in each.
    One image: 784 pixels = 28x28
    returns test, train 
            np.array(8000, 784), np.array(2000, 784)
'''
def getTargetDigit():
    targetdigit_trn = readFile('bindigit_trn.csv')
    targetdigit_tst = readFile('bindigit_tst.csv')
    print(targetdigit_trn.shape, targetdigit_tst.shape)
    return targetdigit_trn, targetdigit_tst

