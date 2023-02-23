import numpy as np

#generate 1D array of floating point numbers
a=1.0*np.arange(0,25,3) 
print('\n My 1D array: \n', a)

#Transform it from 1D array (1x9) to 3x3 square matrix using two methods
b = np.reshape(a, (3, 3)) 
print('\n 1D array reshaped into square matrix using method 1: \n', b)

b = a.reshape((3, 3)) 
print('\n ...and method 2 \n', b)

#Select sequence of elements from an array
print('\n Elements of a from 0th to 7th with the step of 2: \n', a[0:7:2]) 
print('\n All elements of a from 0th to 7th: \n', a[0:7:]) 
print('\n All elements of a from 0th to 7th (method 2): \n', a[:7]) 

##generate a new 3x3 matrix it contains all elements of matrix b, but multiplied by 2.4 
c=2.4*b

#Concatenate them vertically 
d=np.concatenate([b,c])
print('\n Vertically concatenated matrix, method 1: \n', d)
d=np.vstack([b,c])
print('\n ...and method 2: \n', d)

#Concatenate them horizontally
d=np.concatenate([b,c], axis=1)
print('\n Horizontally concatenated matrix, method 1: \n', d)
d=np.hstack([b,c])
print('\n ...and method 2: \n', d)

##Transpose a matrix (swap rows and columns)
d=np.transpose(d)
print('\n Trnsposed matrix: \n', d)

A = np.array([[1, 3, 2], [0, -2, 2], [3, 0, 4]])
print(A)
B = np.delete(A,[2], axis = 1)
print(B)
C = np.array([[1, 2], [0,1]])
D = np.array([[2, 1], [0, 2]])
print(np.concatenate([C, D]))
F = np.arange(0, 16, 1)
print(F)
H = np.arange(0, 16, 4)
print(H)
print(F[:5])
print(F[::5])
print(F[2:3:15])
print(F[0::12])
print(F[7::-1])