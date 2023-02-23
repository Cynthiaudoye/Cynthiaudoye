import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt


x=np.arange(-10,11)*0.3 #Grid of x
y=x**2

plt.figure(figsize=(8,6))

#Add line plot y(x)
plt.plot(x, y, linewidth=2.5, color='black')

#Add scatter plot y(x)
plt.scatter(x, y, color='red')

plt.show()

plt.figure(figsize=(8,6))


##Generate a 2D function - 2D Gaussian - asfunction of x and y
x=np.arange(-10,11)*0.3 #Grid of x
y=x #Grid of y
xx,yy=np.meshgrid(x,y)
zz=np.exp(-xx*xx-yy*yy)

fig,ax=plt.subplots(1,1)
cp = ax.contourf(x,y,zz)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Filled Contours Plot of Z')
#ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

plt.figure(figsize=(8,6))

ax = plt.axes(projection ='3d') 

ax.plot_surface(xx, yy, zz)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

import pandas as pd
mydata = np.genfromtxt('data2.csv', delimiter=' ')

xarr=mydata[:,0]
yarr=mydata[:,1]

print(xarr)
print(yarr)

plt.figure(figsize = (8, 6))
plt.plot(xarr, yarr, linewidth = 2.5, color = 'orange')
plt.show()
