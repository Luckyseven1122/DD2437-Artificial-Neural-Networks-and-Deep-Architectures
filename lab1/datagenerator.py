'''
	Data Generator Module
'''
import sys
import ast
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from iohandler import write_array
from iohandler import read_array
from iohandler import load_data


'''
	Generate a full data set from scratch stored as 
		- [filename]_raw.npy 	
			The raw, sorted dataset. This is useful when applying subsampling.

		- [filename]_labels.npy 	
			The final labels for the dataset.

		- [filename]_inputs.npy 	
			The final dataset activations.
''' 
def generate_binary_data(filename, linear=True, n_points = 50, sA = -1, sB = -1, mAx = -1, mAy = -1, mBx = -1, mBy = -1):
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

	# Save raw dataset before shuffling for subsampling
	write_array(filename+"_raw", x)
	
	return shuffle_data(x)

'''
	Shuffles a raw dataset and separates it into inputs and labels.
'''	
def shuffle_data(x):
	# shuffle columns in x
	inputs = np.zeros([2, x.shape[1]])
	labels = np.zeros([1, x.shape[1]])
	idx = np.random.permutation(x.shape[1])
	for i in idx:
		inputs[:2,i] = x[:2,idx[i]]
		labels[0,i] = x[2,idx[i]]
	labels = labels.astype(int)

	return inputs, labels

'''
	Performs subsampling over a raw dataset and stores the resulting dataset
	as [filename]_cm[class_modifier]_x.npy
'''
def subsample(filename, class_modifier=1):
	'''
	class_modifier = 1: remove random 25% from each class
	class_modifier = 2: remove 50% from classA (labels = -1)
	class_modifier = 3: remove 50% from classB (labels = 1 )
	class_modifier = 4: remove 20% from classA(1,:)<0 (i.e x1 < 0) and
						80% from classA(1,:)>0 (i.e x1 > 0)
	'''
	x = read_array(filename + "_raw")

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

	inputs, labels = shuffle_data(x)
	return inputs, labels

# Simple plot for generated class
def plot_classes(inputs, labels):
    plt.grid(True)
    plt.scatter(inputs[0,:], inputs[1,:], c=labels[0,:])
    plt.show()
    plt.waitforbuttonpress()

def print_logo():
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

def print_menu():
	# Main menu
	menu_choice = input("\nMAIN MENU\n" + 
						 ".~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~._ \n" +
						 "|\n|-1: Generate new dataset \n" +
						 "|\n|-2: Subsample existing dataset \n" +
						 "|\n|-3: Plot existing dataset \n" +
						 "|\n|-Other: Exit program\n" +
						 "|\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~._ \n" +
						 ">")
	menu_choice = ast.literal_eval(menu_choice)
	
	if (menu_choice < 1) or (menu_choice > 3):
		exit()

	return menu_choice

# ------------------ Command line interface ------------------------------

def main():

	quit = False

	while quit == False:

		print_logo()

		menu_choice = print_menu()

		print_logo()

		filename = input("\nEnter target filename --------------------------- \n>")
		
		if menu_choice == 3:

			inputs, labels = load_data(filename)
			plot_classes(inputs,labels)
			mm_query = input("\nReturn to main menu? Y / N \n>")
			quit = False if mm_query == "Y" else True

		if menu_choice == 2:

			# Ask for class modifier
			cm = input(
				"\nChoose class modifier --------------------------- \n" +
				"1: remove random 25% from each class \n" +
				"2: remove 50% from classA (labels = -1) \n" +
				"3: remove 50% from classB (labels = 1 ) \n" +
				"4: remove 20% from classA(1,:)<0 (i.e x1 < 0) and 80% from classA(1,:)>0 (i.e x1 > 0) \n>") 
			cm = ast.literal_eval(cm)
			if (cm < 1) or (cm > 4):
				print("\nInvalid class modifier specified. Exiting.\n")
				mm_query = input("\nReturn to main menu? Y / N \n>")
				quit = False if mm_query == "Y" else True

			# Perform sub-sampling
			inputs, labels = subsample(filename,cm)
			write_array(filename + "_cm" + str(cm) + "_inputs", inputs)
			write_array(filename + "_cm" + str(cm) + "_labels", labels)
			plot_classes(inputs,labels)
			print("\nData written to " + filename + "_cm" + str(cm) + "_inputs.npy and " + filename + "_cm" + str(cm) + "_labels.npy\n")

			mm_query = input("\nReturn to main menu? Y / N \n>")
			quit = False if mm_query == "Y" else True

		if menu_choice == 1:

			# Ask for relevant parameters
			n_points = input("\nEnter number of data points per class ----------- \n>") 
			n_points = ast.literal_eval(n_points)

			cparams = input("\nSet custom parameters for each class? Y / N ----- \n>")
			cparams = True if cparams == "Y" else False

			linear = input("\nUse linear formula for data generation? Y / N ---- \n>")
			linear = True if linear == "Y" else False

			if cparams:
				sA = input("\nEnter sigma for class A (default " + ("0.4)\n>" if linear else "0.3)\n>"))
				sA = ast.literal_eval(sA)
				sB = input("\nEnter sigma for class B (default " + ("0.4)\n>" if linear else "0.3)\n>"))
				sB = ast.literal_eval(sB)
				mAx = input("\nEnter x coordinate for center of class A (default " + ("1.5)\n>" if linear else "1.0)\n>"))
				mAx = ast.literal_eval(mAx)
				mAy = input("\nEnter y coordinate for center of class A (default " + ("0.5)\n>" if linear else "0.3)\n>"))
				mAy = ast.literal_eval(mAy)
				mBx = input("\nEnter x coordinate for center of class B (default " + ("-1.5)\n>" if linear else "0.0)\n>"))
				mBx = ast.literal_eval(mBx)
				mBy = input("\nEnter y coordinate for center of class B (default " + ("-0.5)\n>" if linear else "0.0)\n>"))
				mBy = ast.literal_eval(mBy)

				inputs, labels = generate_binary_data(filename, linear, n_points, sA, sB, mAx, mAy, mBx, mBy)
				write_array(filename + "_inputs", inputs)
				write_array(filename + "_labels", labels)
				plot_classes(inputs,labels)
			else:
				inputs, labels = generate_binary_data(filename, linear, n_points)
				write_array(filename + "_inputs", inputs)
				write_array(filename + "_labels", labels)
				plot_classes(inputs,labels)

			print("\nData written to " + filename + "_inputs.npy and " + filename + "_labels.npy\n")

			mm_query = input("\nReturn to main menu? Y / N \n>")
			quit = False if mm_query == "Y" else True


if __name__ == '__main__':
		main()
