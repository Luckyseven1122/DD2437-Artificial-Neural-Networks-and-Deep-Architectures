'''
	IO Handler for Lab 1: Used to store a generated data set in between runs
'''

import ast
import numpy as np

def write_list(filename, data):
	with open(filename, 'w') as file:  
		file.writelines("%s\n" % d for d in data)

def read_list(filename):
	# define empty list
	data = []

	# open file and read the content in a list
	with open(filename, 'r') as filehandle:  
		filecontents = filehandle.readlines()

		for line in filecontents:
			# remove linebreak which is the last character of the string
			x = line[:-1]

			# parse list
			x = ast.literal_eval(x)

			# add item to the list
			data.append(x)

		# return the contents of the file as a list
		return data

def write_array(filename, data):
	np.save(filename, data)

def read_array(filename):
	return np.load(filename+".npy")


'''
	DEBUG CODE
'''
'''
# Define filename sans .npy-extension
filename = "npout"

# Define dummy array
x=np.array([(1.0,2.1,-3.2), (4.1,5.12,6.2), (7.23,8.0,-9.0)])

# Write array to file
write_array(filename, x)

# Read back array
y = read_array(filename)

# Present result
print(y)
'''
