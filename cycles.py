import numpy as np

a=1.6*np.arange(0,16) ##define an array with 16 elements


#FOR cycle 1
for x in a:
    print(x)
    

#FOR cycle 2
for i in range(0,16,2):
    print(a[i])
    
#WHILE cycle
i=0
while a[i]<10.0:
    print(a[i])
    i=i+1
k = np.arange(0, 11)
print(2 - k)

mydata = np.genfromtxt('data2.csv', delimiter=' ')

xarr=mydata[:,0]
yarr=mydata[:,1]

print(xarr)
print(yarr)

D = 'all element in yarr'
for D in yarr:
    print(D)