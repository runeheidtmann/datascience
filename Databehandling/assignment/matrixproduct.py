# -*- coding: utf-8 -*-

# =============================================================================
# A and B are matrices with sizes MxN and RxK.
# The product of A and B is defined if and only if N = R.
# If N = R then the product AB is a matrix of size MxK. 
# =============================================================================

# =============================================================================
# A = [[a00,b01],
#      [c10,d11]], 
#
# B = [[b00,b01],
#      [b10,11]]
# 
# ----> C[i][j] = ai1*b1j+ai2*b2j ... Ain*Bnj  
# =============================================================================

# =============================================================================
# We will write a method that takes to matrices and calculates the product.
#
# Checklist:
# [] Found out if the product of the to given matrices is defined.
#       [] Are A and B valid matrices?
#       [] Are N = R?
# [] Calculate and return product.
# =============================================================================

def multiply_matrices(a,b):
    #is the product defined?
    size_a = matrixSize(a)
    size_b = matrixSize(b)
    
    if is_defined(size_a, size_b):
        return get_product(a,b)
    
    return None

def get_product(a,b):
    product = [[0]*len(b[0]) for i in range(len(a))]
    
    for i in range(len(product)):
        for j in range(len(product[0])):
            for n in range(len(b)):
                product[i][j] += a[i][n]*b[n][j] 
    
    return product
    
def is_defined(a,b):
   if a and b is not None:    
       return a[1] == b[0]
   return False

    
def matrixSize(matrix):
    
    rows = None
    cols = None
    len_of_row = len(matrix[0])
    
    for row in matrix:
        if len(row) != len_of_row:
            return None
    
    cols = len(matrix[0])
    rows = len(matrix)
    
    return [rows,cols]

def transpose(matrix):
    matrix_size = matrixSize(matrix)
    transpose = [[0]*matrix_size[0] for i in range(matrix_size[1])]
    
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            transpose[j][i] = matrix[i][j] 
    
    return transpose    

def matrixAddition(a,b):
    if matrixSize(a) != matrixSize(b):
        return None
    
    #Give result matrix C the right size:
    #List comprehensions makes lists
    c = [[0]*len(a[0]) for i in range(len(a))]
    
    for i in range(len(c)):
        for j in range(len(c[0])):
            c[i][j] = a[i][j]+b[i][j]
    
    return c

def scalaMultiplikation(scala,matrix):
    
    resultMatrix = [[0]*len(matrix[0]) for i in range(len(matrix))]
    
    for i in range(len(resultMatrix)):
        for j in range(len(resultMatrix[0])):
            resultMatrix[i][j] = scala*matrix[i][j]
    
    return resultMatrix
    
def inverse(a):
    #takes a 2x2 matrix:
    a_size = matrixSize(a)
    if a_size[0] != 2 or a_size[1] != 2:
        return False
    
    determinant = a[0][0]*a[1][1]-a[0][1]*a[1][0]
    flippedMatrix = [[a[1][1], -a[0][1]],
                     [-a[1][0], a[0][0]]
                     ]
    return scalaMultiplikation(1/determinant, flippedMatrix)


    
    


k = [[1,2],[4,5]]
m = [[1,2,3],[4,5,6]]
print(multiply_matrices(k, inverse(k)))




