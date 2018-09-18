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


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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
	print(bcolors.HEADER + bcolors.BOLD + "\n\n_______ _________ ______   _______    ______   _______  _______  _______\n" + 
	"(  ____ \\__   __/(  __  \ (  ____ \  (  ___ \ (  ____ )(  ___  )(  ____ )\n" +
	"| (    \/   ) (   | (  \  )| (    \/  | (   ) )| (    )|| (   ) || (    )|\n" +
	"| (__       | |   | |   ) || (__      | (__/ / | (____)|| |   | || (____)|\n" +
	"|  __)      | |   | |   | ||  __)     |  __ (  |     __)| |   | ||     __)\n" +
	"| (         | |   | |   ) || (        | (  \ \ | (\ (   | |   | || (\ (   \n" +
	"| (____/\___) (___| (__/  )| (____/\  | )___) )| ) \ \__| (___) || ) \ \__\n" +
	"(_______/\_______/(______/ (_______/  |/ \___/ |/   \__/(_______)|/   \__/\n" + bcolors.UNDERLINE +
	"Ultimate Data Generator 1.9 Flex Edition XP\n" + 
	"Presented by Google, Microsoft and Uber\n\n" + bcolors.ENDC)

def print_menu():
	# Main menu
	menu_choice = input(bcolors.BOLD + "\nMAIN MENU\n" + bcolors.ENDC +
						 ".~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~._ \n" +
						 "|\n|-1: Generate new dataset \n" +
						 "|\n|-2: Subsample existing dataset \n" +
						 "|\n|-3: Plot existing dataset \n" +
						 "|\n|-4: Set symmetric / asymmetric labels \n"
						 "|\n|-Other: Exit program\n" +
						 "|\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~._ \n" +
						 ">")

	menu_choice = check_numeric(menu_choice)
	
	if (menu_choice < 1) or (menu_choice > 4):
		exit()

	return menu_choice

def return_to_menu():
	mm_query = input(pad("Return to main menu? Y / N ", True)).lower().strip()
	mm_query = check_yes_no(mm_query)
	return mm_query

def is_float(x):
	try:
		float(x)
	except ValueError:
		return False

	return True

def check_numeric(x):
	while not x.isnumeric() and not is_float(x):
		x = input(pad("Please specify a numeric value ", True))
	return ast.literal_eval(x)

def check_yes_no(x):
	while x.lower().strip() != "y" and x.lower().strip() != "n":
		x = input(pad("Please make your selection: Y / N ", True))
	return True if x.lower().strip() == "y" else False

def pad(x, prompt):
	return "\n" + x.ljust(60, "-") + "._\n" + (">" if prompt else "")


# ------------------ Command line interface ------------------------------

def main():

	quit = False

	while quit == False:

		print_logo()

		menu_choice = print_menu()

		print_logo()

		filename = input(pad("Enter target filename ", True))

		if menu_choice == 4:
			inputs, labels = load_data(filename)

			symmetric = check_yes_no(input(pad("Use symmetric labels? Y / N ", True)))

			if symmetric:
				labels = np.where(labels == 0, -1, labels)
			else:
				labels = np.where(labels == -1, 0, labels)

			filename = filename + ("_symmetric" if symmetric else "_asymmetric")			
			write_array(filename + "_inputs", inputs)
			write_array(filename + "_labels", labels)

			print("\nData written to " + filename + "_inputs.npy and " + filename + "_labels.npy\n")
		
		if menu_choice == 3:

			inputs, labels = load_data(filename)
			plot_classes(inputs,labels)

		if menu_choice == 2:

			# Ask for class modifier
			cm = check_numeric(input(
				pad("Choose class modifier ", False) +
				"1: remove random 25% from each class \n" +
				"2: remove 50% from classA (labels = -1) \n" +
				"3: remove 50% from classB (labels = 1 ) \n" +
				"4: remove 20% from classA(1,:)<0 (i.e x1 < 0) and 80% from classA(1,:)>0 (i.e x1 > 0) \n>"))

			while cm < 1 or cm > 4:
				cm = check_numeric(input(pad("Please enter a value between 1 and 4 ", True)))

			# Perform sub-sampling
			inputs, labels = subsample(filename,cm)
			write_array(filename + "_cm" + str(cm) + "_inputs", inputs)
			write_array(filename + "_cm" + str(cm) + "_labels", labels)
			plot_classes(inputs,labels)
			print("\nData written to " + filename + "_cm" + str(cm) + "_inputs.npy and " + filename + "_cm" + str(cm) + "_labels.npy\n")

		if menu_choice == 1:

			# Ask for relevant parameters
			n_points = check_numeric(input(pad("Enter number of data points per class ", True)))

			cparams = check_yes_no(input(pad("Set custom parameters for each class? Y / N ", True)))

			linear = check_yes_no(input(pad("Use linear formula for data generation? Y / N ", True)))

			if cparams:
				sA = check_numeric(input(pad("Enter sigma for class A (default " + ("0.4) " if linear else "0.3) "), True)))
				sB = check_numeric(input(pad("Enter sigma for class B (default " + ("0.4) " if linear else "0.3) "), True)))
				mAx = check_numeric(input(pad("Enter x coordinate for center of class A (default " + ("1.5) " if linear else "1.0) "), True)))
				mAy = check_numeric(input(pad("Enter y coordinate for center of class A (default " + ("0.5) " if linear else "0.3) "), True)))
				mBx = check_numeric(input(pad("Enter x coordinate for center of class B (default " + ("-1.5) " if linear else "0.0)\ "), True)))
				mBy = check_numeric(input(pad("Enter y coordinate for center of class B (default " + ("-0.5) " if linear else "0.0) "), True)))

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

		quit = not return_to_menu()


if __name__ == '__main__':
		main()
