# Library of Common Matrix Operations 
# These will be applied to various Matrix based problems
# and Images
import numpy as np
from numpy import asarray
from math import pi, cos, sin
from itertools import permutations

# returns a square matrix for given dimension n and factor alpha
# alpha != 1 can be used for uniform scaling, alpha = 1 gives identity matrix I
def id_mat(n,alpha):
    I = [[0 for x in range(n)] for x in range(n)]
    for i in range(n):
        I[i][i] = alpha
    return I


# transpose of given matrix A
def transpose_mat(A):
    m = len(A)
    n = len(A[0])
    return [[A[j][i] for j in range(m)] for i in range(n)]


# Prints an nD array or npArray in more concise format
# this is not an explicit matrix operation, just an accessory to the library
def print_matrix(A):
    n = len(A[0])
    i = 1

    print(f"  |{[x for x in range(1,n+1)]}|")
    for row in A:
        print(f"|{i}|{row}|")
        i = i + 1


# create a square matrix based on scale factor
# (if scaling factor = n, then original dimensions of matrix are scaled by 1/n)
# can be used to shrink an image's dimensions and increase speed of operations performed
def scaled_mat(A,scale_factor):
	scaled_columns = [row[::scale_factor] for row in A]
	return asarray([scaled_columns[i] for i in range(0, len(A), scale_factor)])


# check for valid dimensions and implement matrix-matrix multiplication
def matmat(A , B):
    mA , nA = len(A), len(A[0])
    mB , nB = len(B), len(B[0])
    if nA == mB:
        C = [[0 for x in range(nB)] for x in range(mA)]
        for i in range(mA):
            for j in range(nB):
                row = [A[i][k] for k in range(mB)]
                col = [B[k][j] for k in range(mB)]
                C[i][j] = sum([row[i]*col[i] for i in range(len(row))])
        return asarray(C)
    else:
        print('Invalid Dimensions for Matrix-Matrix Multiplication')


# check for valid dimensions and implement matrix-vector multiplication
def matvec(A , v):
    mA , nA = len(A), len(A[0])
    m = len(v)
    if nA == m:
        D = [0 for x in range(mA)]
        for i in range(mA):
            D[i] = sum([A[i][j]*v[j] for j in range(m)])
        return asarray(D)
    else:
        print('Invalid Dimensions for Matrix-Vector Multiplication')


# rotation matrix for two dimensional applications
def rotation_mat_2d(theta):
	return[[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]


# rotation matrix for three dimensional applications and specified axis of rotation
def rotation_mat_3d(axis,theta):
	if axis == 'x':
		return[[1,0,0],[0,cos(theta), -sin(theta)], [0,sin(theta), cos(theta)]]
	elif axis == 'y':
		return[[cos(theta),0, sin(theta)],[0,1,0],[-sin(theta),0,cos(theta)]]
	elif axis == 'z':
		return[[cos(theta), -sin(theta), 0], [sin(theta), cos(theta),0],[0,0,1]]
	else:
		print('Incorect Axis Given!')

	
# triangle_index finds if given matrix containts a triangular configuration and specifies 
# which indices yeild the potential configuration
def triangle_index(A):

    if len(A) == len(A[0]):
        indices = [x for x in range(len(A))]
        perms = list(permutations(indices))

        for p in perms:
            perm_A = [A[p[i]] for i in range(len(p))]
            # gather and check diagonal entries in A to be nonzero
            diags_left = [perm_A[i][i] for i in range(len(A))]
            diags_right = [perm_A[i][(len(A)-1)-i] for i in (range(len(A)))]

            # check upper triangular left / lower triangular left
            if all(diags_left) == True:
                below_diags_left = [perm_A[row][col] for row in range(1, len(perm_A))
                                    for col in range(0,len(perm_A)-1) if col != row]
                above_diags_right = [perm_A[row][col] for row in range(0, len(perm_A)-1)
                                     for col in range(1,len(perm_A)) if col != row]
                if all(x == 0 for x in below_diags_left):
                    print('Upper Right Triangular Configuration Exists')
                    print(f'Row Indices : {p}')
                    return p

                elif all(x == 0 for x in above_diags_right):
                    print('Lower Right Triangular Configuration Exists')
                    print(f'Row Indices : {p}')
                    return p

                else:
                    return 'No Triangular Configurations Exist'

            # check upper triangular right / lower triangular right
            elif all(diags_right) == True:
                below_diags_right = [perm_A[row][col] for row in range(1, len(perm_A))
                                     for col in range(1, len(perm_A)) if col != row]
                above_diags_left = [perm_A[row][col] for row in range(0, len(perm_A)-1)
                                    for col in range(0, len(perm_A)-1) if col != row]
                if all(x == 0 for x in below_diags_right):
                    print('Upper Left Configuration Triangular Exists')
                    print(f'Row Indices : {p}')
                    return p

                elif all(x == 0 for x in above_diags_left):
                    print('Lower Left Triangular Configuration Exists')
                    print(f'Row Indices : {p}')
                    return p

                else:
                    print('No Triangular Configurations Exist')

    else:
        print('Incorrect Dimensions!')







