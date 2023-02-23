import numpy as np

#
#Create four floating point variables a, b, c, and d
a=2.0
b=4.4
c=-1.5
d=3.2

#Calculate and print u
u=3.2*(a+b)/c+d**3.0
print('Answer:', u)



#
#Create two arrays
a=[1.0, 0.0, 2.0]
b=[-1.0, 1.0, -0.5]

#Calculate their dot-product and print the result
u=a[0]*b[0]+a[1]*b[1]+a[2]*b[2] ## direct method
v=np.dot(a,b) ## using numpy library

print('Answers are', u, ' and ', v)


#
#Define two matrices
a=np.matrix([[0,1,2],[0,1,3],[4,4,4]])
b=np.matrix([[2,1,0],[2,1,-1],[-2,-2,-2]])

#print them
print(a)
print(b)

#add them and print the result
u=a+b
print('Answer:')
print(u)

A = np.array([4, 2, 0])
B = np.array([0, 2, 4])
X = np.sum(A + B)
Y = np.subtract(A, B)
print(X)
print(Y)

D = np.array([[1, 2], [0, 1]])
E = np.array([[2, 1], [0, 2]])
C = (4 * A) - B
print(C)

F = np.array([[1, 2, 3], [1, -2, -3], [1, 2, -3]])
T = np.sum(F)
print(T)