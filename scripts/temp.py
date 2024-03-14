
import pandas as pd  
import numpy as np 

data=pd.read_hdf('/mnt/raptor/hassan/data/voc/vocdevkit/voc2007/ImageSets/Main/test_labels_AT.h5',key='df').iloc[:,:-1].to_numpy()
print(data.shape)

n=data.shape[1]
s=2**np.arange(n)
x1D=data.dot(s)
Xout=(x1D[:,None]==x1D).astype(float)
print(Xout)
print(Xout.shape)
sumval=np.sum(Xout,axis=1)
sortedidx=np.argsort(sumval)[::-1]

for idx in sortedidx[:460]:
	print(idx,data[idx,:])

print(data[:10])
print('n:',data[sortedidx[460]])

data2=np.abs(data-data[sortedidx[460]])
print(data2[:10])
0/0
x1D=data2.dot(s)
sortedval=np.sort(x1D)
print(sortedval)
a=np.argmin(sortedval>0)
print(list(sortedval[a:a+50]))
print(data[a:a+50])
#print(data[sortedidx[460]])




0/0


####################################

names=['hin',
'stride',
'padding', 
'dilation',
'kernel',
'output_padding']

# def calc(arr):
# 	#Hout = (hin - 1)*stride - 2*padding + dilation * (kernel - 1) + output_padding + 1
# 	return (arr[0] - 1)*arr[1] - 2*arr[2] + arr[3] * (arr[4] - 1) + arr[5] + 1
def calc(arr):
	#Hout = (hin - 1)*stride - 2*padding + dilation * (kernel - 1) + output_padding + 1
	return (arr[0] + 2 * arr[2] - arr[3]*(arr[4]-1) - 1)/arr[1] + 1
	#return (arr[0] - 1)*arr[1] - 2*arr[2] + arr[3] * (arr[4] - 1) + arr[5] + 1


arrs=[
]

for strides in [1,2,3]:
	for padding in [0,1]:
		for kernels in [3,5,7]:		
			arrs.append([224,strides,padding,1,kernels,0])
	
#1 7 11 27 55 111 224
#211, 213, 215, 209, 
for a in arrs:
	print(', '.join([names[t]+':'+str(a[t]) for t in range(6)]),' - ', calc(a))




#hin:3, stride:1, padding:0, dilation:1, kernel:5, output_padding:0  -  7
# 7 -> 13, 15
# 13 -> 27, 29
# 15 -> 31 33 35
# 27-> 55 57 59
# 29 -> 59 61 63
# 31 -> 61 63 65 67
# 33 -> 65 67 69 71
# 35 -> 69, 71, 73, 75

# 55-> 111, 113

# 111 -> 224 hin:111, stride:2, padding:0, dilation:1, kernel:3, output_padding:1