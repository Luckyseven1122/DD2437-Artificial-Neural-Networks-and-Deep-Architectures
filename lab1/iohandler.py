'''
	IO Handler for Lab 1: Used to store a generated data set in between runs
'''

import ast

def write_data(filename, data):
	with open(filename, 'w') as file:  
		file.writelines("%s\n" % d for d in data)

def read_data(filename):
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

'''
	DEBUG CODE
'''
'''
# Define filename
filename = "testfil.txt"

# Define dummy list
my_list=[[1.0,2.1,-3.2], [4.1,5.12,6.2], [7.23,8.0,-9.0]]

# Write list to file
write_data(filename, my_list)

# Read back list
my_list = read_data(filename)

# Present result
print(my_list)
'''
