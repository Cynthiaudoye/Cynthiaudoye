import numpy as np

k=4 #int variable

a=6.28 # float variable

aostring=['word0', 'word1', 'word2'] #set of strings

print('\n My set of string variables:', aostring)

aoi=np.arange(0,16,k) ##array of integer numbers from 0 to 16, with the step equal k

print('\n My sequence of numbers:', aoi)

aof=np.random.randint(8, size=([3,3])) ##3x3 matrix of random numbers

print('\n Matrix of randomly generated numbers:\n', aof)

A = np.array([-2, 1, 0, 1, 2])
print(A)

a = 1; b = 2; c = 3; d = 0; e = 2; f = 3; g = 0; h = 0; k = 3
X = np.matrix([[a, b, c],[d, e, f], [g, h, k]])
print(X)

x = 'Hello'
y = ','
z = 'world'
Y = np.array([x, y, z])
print(Y)
