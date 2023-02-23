import numpy as np
import pandas as pd

#Read CSV using panda
mydata = pd.read_csv('data1.csv', sep=' ')
#Convert columns into lists
mdnum=mydata['Number'].tolist()
mdval=mydata['Value'].tolist()
mdcol=mydata['Colour'].tolist()
#Convert 'Number' and 'Value' columns into Numpy arrays
mdnum=np.array(mdnum)
mdval=np.array(mdval)
print('\n Output of method 1:')
print(mdnum)
print(mdval)
print(mdcol)



#Read CSV using numpy
mydata = np.genfromtxt('data2.csv', delimiter=' ')
#Use first column of mydata as xarr array, second column as yarr array
xarr=mydata[:,0]
yarr=mydata[:,1]

print(' \n Output of method 2:')
print(xarr)
print(yarr)

data2 = pd.read_csv('data2.csv')