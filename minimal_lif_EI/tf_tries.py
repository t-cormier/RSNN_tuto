import tensorflow as tf
import numpy as np

mat1 = np.ones((4,5), dtype = bool)
print(type(mat1))
mat2 = np.zeros((4,2), dtype = bool)
mat12 = np.concatenate((mat1,mat2), axis=1)
print(mat12)
