'''
	Data Generator Module
'''
import sys
import ast
import os
import numpy as np
import math
from iohandler import write_array

def generate_binary_data(linear=True, class_modifier=0, n_points = 50, sA = -1, sB = -1, mAx = -1, mAy = -1, mBx = -1, mBy = -1):
    '''
    class_modifier = 0: no subsampling
    class_modifier = 1: remove random 25% from each class
    class_modifier = 2: remove 50% from classA (labels = -1)
    class_modifier = 3: remove 50% from classB (labels = 1 )
    class_modifier = 4: remove 20% from classA(1,:)<0 (i.e x1 < 0) and
                        80% from classA(1,:)>0 (i.e x1 > 0)
    '''

    if linear:
        '''
        Generates two linearly separable classes of points
        Note: axis are set between -3 and 3 on both axis
        Note: Labels (-1, 1)
        '''

        mA = np.array([ 1.5 if mAx == -1 else mAx, 0.5 if mAy == -1 else mAy])
        mB = np.array([-1.5 if mBx == -1 else mBx, -0.5 if mBy == -1 else mBy])
        sigmaA = 0.4 if sA == -1 else sA
        sigmaB = 0.4 if sB == -1 else sB

        x = np.zeros([3, n_points*2])
        x[0,:n_points] = np.random.randn(1, n_points) * sigmaA + mA[0]
        x[1,:n_points] = np.random.randn(1, n_points) * sigmaA + mA[1]

    else:
        '''
        Generates two non-linearly separable classes of points
        '''
        mA = [ 1.0 if mAx == -1 else mAx, 0.3 if mAy == -1 else mAy]
        mB = [ 0.0 if mBx == -1 else mBx, 0.0 if mBy == -1 else mBy]
        sigmaA = 0.3 if sA == -1 else sA
        sigmaB = 0.3 if sB == -1 else sB

        x = np.zeros([3, n_points*2])
        x[0,:math.floor(n_points/2)] = np.random.randn(1, math.floor(n_points/2)) * sigmaA - mA[0]
        x[0,math.floor(n_points/2):n_points] = np.random.randn(1, math.floor(n_points/2)) * sigmaA + mA[0]
        x[1,:n_points] = np.random.randn(1, n_points) * sigmaA + mA[1]

    x[2,:n_points] = -1
    x[0,n_points:] = np.random.randn(1, n_points) * sigmaB + mB[0]
    x[1,n_points:] = np.random.randn(1, n_points) * sigmaB + mB[1]
    x[2,n_points:] = 1

    if class_modifier == 1:
        idx = np.arange(math.floor(x.shape[1]/2))
        idxA = idx[:math.floor(x.shape[1]/4)]
        idxB = idx[math.floor(x.shape[1]/4):] # +1 ???
        np.random.shuffle(idxA)
        np.random.shuffle(idxB)
        idxA = idxA[:math.floor(x.shape[1]/8)]
        idxB = idxB[:math.floor(x.shape[1]/8)]
        idx = np.concatenate((idxA, idxB))
        x = np.delete(x, idx, axis=1)
    if class_modifier == 2:
        idx = np.arange(math.floor(x.shape[1]/2))
        np.random.shuffle(idx)
        idx = idx[:math.floor(x.shape[1]/4)]
        x = np.delete(x, idx, axis=1)
    if class_modifier == 3:
        idx = np.arange(math.floor(x.shape[1]/2),x.shape[1])
        np.random.shuffle(idx)
        idx = idx[:math.floor(x.shape[1]/4)]
        x = np.delete(x, idx, axis=1)
    if class_modifier == 4:
        idx = np.arange(math.floor(x.shape[1]/2))
        classA = x[:,idx]
        xless = np.where(classA[0,:] < 0, idx, -1)
        xmore = np.where(classA[0,:] >= 0, idx, -1)
        xless = xless[xless >= 0]
        xmore = xmore[xmore >= 0]
        np.random.shuffle(xless)
        np.random.shuffle(xmore)
        xless_size = xless.shape[0]
        xmore_size = xmore.shape[0]
        xless = xless[:math.floor(x.shape[1]*0.8)]
        xmore = xmore[:math.floor(x.shape[1]*0.2)]
        idx = np.concatenate((xless, xmore))
        x = np.delete(x, idx, axis=1)
    # shuffle columns in x
    inputs = np.zeros([2, x.shape[1]])
    labels = np.zeros([1, x.shape[1]])
    idx = np.random.permutation(x.shape[1])
    for i in idx:
        inputs[:2,i] = x[:2,idx[i]]
        labels[0,i] = x[2,idx[i]]
    labels = labels.astype(int)

    return inputs, labels

# ------------------ Command line interface ------------------------------


# Clear terminal window
os.system('cls' if os.name == 'nt' else 'clear')

# Print logo
print("\n\n_______ _________ ______   _______    ______   _______  _______  _______\n" + 
"(  ____ \\__   __/(  __  \ (  ____ \  (  ___ \ (  ____ )(  ___  )(  ____ )\n" +
"| (    \/   ) (   | (  \  )| (    \/  | (   ) )| (    )|| (   ) || (    )|\n" +
"| (__       | |   | |   ) || (__      | (__/ / | (____)|| |   | || (____)|\n" +
"|  __)      | |   | |   | ||  __)     |  __ (  |     __)| |   | ||     __)\n" +
"| (         | |   | |   ) || (        | (  \ \ | (\ (   | |   | || (\ (   \n" +
"| (____/\___) (___| (__/  )| (____/\  | )___) )| ) \ \__| (___) || ) \ \__\n" +
"(_______/\_______/(______/ (_______/  |/ \___/ |/   \__/(_______)|/   \__/\n" +
"Ultimate Data Generator 1.9 Flex Edition XP\n" + 
"Presented by Google, Microsoft and Uber\n\n")

# Ask for relevant parameters
filename = input("\nEnter target filename --------------------------- \n>")

n_points = input("\nEnter number of data points per class ----------- \n>") 
n_points = ast.literal_eval(n_points)

cm = input(
	"\nChoose class modifier --------------------------- \n" +
 	"0: no subsampling \n" + 
 	"1: remove random 25% from each class \n" +
 	"2: remove 50% from classA (labels = -1) \n" +
 	"3: remove 50% from classB (labels = 1 ) \n" +
 	"4: remove 20% from classA(1,:)<0 (i.e x1 < 0) and 80% from classA(1,:)>0 (i.e x1 > 0) \n>") 
cm = ast.literal_eval(cm)

cparams = input("\nSet custom parameters for each class? Y / N ----- \n>")
cparams = True if cparams == "Y" else False

linear = input("\nUse linear formula for data generation? Y / N ---- \n>")
linear = True if linear == "Y" else False

if cparams:
    sA = input("\nEnter sigma for class A (default " + "0.4)\n>" if linear else "0.3)\n>")
    sA = ast.literal_eval(sA)
    sB = input("\nEnter sigma for class B (default " + "0.4)\n>" if linear else "0.3)\n>")
    sB = ast.literal_eval(sB)
    mAx = input("\nEnter x coordinate for center of class A (default " + "1.5)\n>" if linear else "1.0)\n>")
    mAx = ast.literal_eval(mAx)
    mAy = input("\nEnter y coordinate for center of class A (default " + "0.5)\n>" if linear else "0.3)\n>")
    mAy = ast.literal_eval(mAy)
    mBx = input("\nEnter x coordinate for center of class B (default " + "-1.5)\n>" if linear else "0.0)\n>")
    mBx = ast.literal_eval(mBx)
    mBy = input("\nEnter y coordinate for center of class B (default " + "-0.5)\n>" if linear else "0.0)\n>")
    mBy = ast.literal_eval(mBy)

    inputs, labels = generate_binary_data(linear,cm,n_points, sA, sB, mAx, mAy, mBx, mBy)
    write_array(filename + "_inputs", inputs)
    write_array(filename + "_labels", labels)
else:
    inputs, labels = generate_binary_data(linear,cm,n_points)
    write_array(filename + "_inputs", inputs)
    write_array(filename + "_labels", labels)

print("\nData written to " + filename + "_inputs.npy and " + filename + "_labels.npy\n")

