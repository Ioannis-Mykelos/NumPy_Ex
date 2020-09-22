#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
print(np.__version__)
#1 Create a null vector of size 10.
z=np.zeros(10)
print(z)

#2 Create a null vector of size 10 but the fifth value is 40.
m=np.zeros(10)
m[4]=40
print(m)

#3 Create a vector with values ranging from 10 to 49, with lenght = 40.
n=np.arange(10,50)
print(len(n))
print(n)

#4 Reverse the vector n (first element becomes last).
n=n[::-1]
print(n)

#5 Create a 3x3 matrix with values ranging from 0 to 8.
z= np.arange(9).reshape(3,3)
print(z)

#6 Reverse the lines of the matrix z (first line becomes last).
z=z[::-1]
print(z)

#7 Create a 3x3 matrix with values ranging from 0 to 8 , but reverse it.
z= np.arange(9).reshape(3,3)
l=[]
for i in z:
    i=i[::-1]
    l.append(i)
    print(i)
k=np.array(l)
k=k[::-1]
print(k)

#8 Find indices of nonzero elements from a list.
nz = np.nonzero([1,2,0,0,4,0])
print(nz)

#9 Create a 3x3 identity matrix of integers.
m = np.eye(3, dtype=int)
print(m)

#10 Create a 3x3 array with random values.
m = np.random.random((3,3))
print(m)

#11 Create a 10x10 array with random values and find the minimum and maximum values.
Z = np.random.random((10,10))
Zmin, Zmax = Z.min(), Z.max()
print(Zmin, Zmax)

#12 Create a random vector of size 30 and find the mean value.
z= np.random.random(30)
m = z.mean()
print(m)

#13 Create a 2d array with 1 on the border and 0 inside.
z = np.ones((10,10), dtype=int)
z[1:-1,1:-1] = 0
print(z)

#14 Create a 5x5 matrix with values 1,2,3,4 just below the diagonal.
z = np.diag(1+np.arange(4),k=-1)
print(z)

#15 Create a 8x8 matrix and fill it with a checkerboard pattern.
z = np.zeros((8,8),dtype=int)
z[1::2,::2] = 1
z[::2,1::2] = 1
print(z)

#16 Create a checkerboard 8x8 matrix using the tile function.
Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
print(Z)

#17 Normalize a 5x5 random matrix.
Z = np.random.random((5,5))
Zmax, Zmin = Z.max(), Z.min()
Z = (Z - Zmin)/(Zmax - Zmin)
print(Z)

#18 Create a custom dtype that describes a color as four unisgned bytes.
color = np.dtype([("r", np.ubyte, 1),
("g", np.ubyte, 1),
("b", np.ubyte, 1),
("a", np.ubyte, 1)])
print(color)

#19 Multiply a 5x3 matrix by a 3x2 matrix (real matrix product).
Z = np.dot(np.ones((5,3)), np.ones((3,2)))
print(Z)

#20 Given a 1D array, negate all elements which are between 3 and 8, in place.
Z = np.arange(11)
Z[(3 < Z) & (Z <= 8)] *= -1
print(Z)

#21 Extract the integer part of a random array using 5 different methods.
Z = np.random.uniform(0,10,10)
print (Z - Z%1)
print (np.floor(Z))
print (np.ceil(Z)-1)
print (Z.astype(int))
print (np.trunc(Z))

#22 Create a 5x5 matrix with row values ranging from 0 to 4.
Z = np.zeros((5,5),dtype=int)
Z += np.arange(5)
print(Z)

#23 Consider a generator function that generates 10 integers and use it to build an array.
def generate():
    for x in range(10):
        yield x
z = np.fromiter(generate(),dtype=float,count=-1)
print(z)

#24 Create a vector of size 10 with values ranging from 0 to 1, both excluded.
z = np.linspace(0,1,12,endpoint=True)[1:-1]
print(z)

#25 Create a random vector of size 10 and sort it.
z = np.random.random(10)
z.sort()
print(z)

#26 Consider two random array A anb B, check if they are equal.
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)
equal = np.allclose(A,B)
print(equal)

#27 Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates.
Z = np.random.random((10,2))
X,Y = Z[:,0], Z[:,1]
R = np.sqrt(X**2+Y**2)
T = np.arctan2(Y,X)
print(R)
print(T)

#28 Create random vector of size 10 and replace the maximum value by 0.
Z = np.random.random(10)
Z[Z.argmax()] = 0
print(Z)

#29 Create a structured array with x and y coordinates covering the [0,1]x[0,1] area.
Z = np.zeros((10,10), [('x',float),('y',float)])
Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,10),np.linspace(0,1,10))
print(Z)

#30 Given two arrays, X and Y, construct the Cauchy matrix C (Cij = 1/(xi yj)).
X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X, Y)
print(np.linalg.det(C))

#31 Print all the values of an array.
np.set_printoptions(threshold=np.nan)
Z = np.zeros((25,25))
print(Z)

#32 Find the closest value (to a given scalar) in an array.
Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z[index])

#33 Consider a random vector with shape (100,2) representing coordinates, find point by point distances.
Z = np.random.random((10,2))
X,Y = np.atleast_2d(Z[:,0]), np.atleast_2d(Z[:,1])
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(D)
# Much faster with scipy
import scipy
import scipy.spatial
Z = np.random.random((10,2))
D = scipy.spatial.distance.cdist(Z,Z)
print(D)

#34 convert a float (32 bits) array into an integer (32 bits) in place.
Z = np.arange(10, dtype=np.int32)
Z = Z.astype(np.float32, copy=False)

#35 Generate a generic 2D Gaussianlike array.
X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
print(G)

#36 Randomly place p elements in a 2D array.
n = 10
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1)

#37 Subtract the mean of each row of a matrix.
X = np.random.rand(5, 10)
Y = X - X.mean(axis=1, keepdims=True)
print(Y)

#38 Find the nearest value from a given value in an array.
Z = np.random.uniform(0,1,10)
z = 0.5
m = Z.flat[np.abs(Z - z).argmin()]
print(m)

#39 Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices).
Z = np.ones(10)
I = np.random.randint(0,len(Z),20)
Z += np.bincount(I, minlength=len(Z))
print(Z)

#40 Accumulate elements of a vector (X) to an array (F) based on an index list (I).
X = [1,2,3,4,5,6]
I = [1,3,9,3,4,1]
F = np.bincount(I,X)
print(F)

#41 Consider the vector [1, 2, 3, 4, 5], build a new vector with 3 consecutive zeros interleaved between each value.
Z = np.array([1,2,3,4,5])
nz = 3
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
Z0[::nz+1] = Z
print(Z0)

#42 Consider an array of dimension (5,5,3), mulitply it by an array with dimensions (5,5).
A = np.ones((5,5,3))
B = 2*np.ones((5,5))
print(A * B[:,:,None])

#43 Swap two rows of an array.
A = np.arange(25).reshape(5,5)
A[[0,1]] = A[[1,0]]
print(A)

#43 Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]. 
#   Generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]].
Z = np.arange(1,15,dtype=uint32)
R = stride_tricks.as_strided(Z,(11,4),(4,4))
print(R)

#44 Compute a matrix rank.
Z = np.random.uniform(0,1,(10,10))
U, S, V = np.linalg.svd(Z) # Singular Value Decomposition
rank = np.sum(S > 1e-10)

#45 Find the most frequent value in an array.
Z = np.random.randint(0,10,50)
print(np.bincount(Z).argmax())

#46 Extract all the contiguous 3x3 blocks from a random 10x10 matrix.
Z = np.random.randint(0,5,(10,10))
n = 3
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print(C)

#47 Get the n largest values of an array.
Z = np.arange(10000)
np.random.shuffle(Z)
n = 5
print(Z[np.argpartition(-Z,n)[:n]])

#48 Given an arbitrary number of vectors, build the cartesian product (every combinations of every item).
def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays]
    shape = (len(x) for x in arrays)
    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T
    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]
    return ix
print (cartesian(([1, 2, 3], [4, 5], [6, 7])))

#49 Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]).
Z = np.random.randint(0,5,(10,3))
E = np.logical_and.reduce(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print(Z)
print(U)

#50 










# In[ ]:




