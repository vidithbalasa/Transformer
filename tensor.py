'''
Apparently all you need for a tensor is a list with a couple helper functions.
'''

import math
from micrograd.engine import Value


def shape(l: list) -> tuple:
    '''
    Recursively returns the shape of an n-dimensional list.
    '''
    if not isinstance(l[0], list):
        return (len(l),)
    return (len(l),) + shape(l[0])

def flatten_nd(l: list) -> list:
    '''
    Flatten an n-dimentional list.
    '''
    x = l.copy()
    while isinstance(x[0], list):
        x = [item for sublist in x for item in sublist]
    return x

def reshape_flattened_nd(l: list, og_shape: tuple) -> list:
    '''
    Reshape a flattened list into an n-dimensional list.
    '''
    matrix = l.copy()
    if shape(matrix) == og_shape:
        return matrix
    if len(og_shape) == 1:
        return matrix
    splits = math.ceil(len(matrix) / og_shape[0])
    return [reshape_flattened_nd(matrix[x:x+splits], og_shape[1:]) for x in range(0, len(matrix), splits)]

def matmul_4d(a: list, b: list) -> list:
    '''
    Matrix multiplication of two 4-dimensional tensors.

    Matrix math from "Multidimensional Matrix Math" by Ashu M. G. Solo:
    http://www.iaeng.org/publication/WCE2010/WCE2010_pp1829-1833.pdf

    Requirements
    --------------------------------------------------------------
    1. A_y == B_x
    2. Every dimension above y is the same length for both matrices (e.g. A_i == B_i for all i > y)

    Returns
    --------------------------------------------------------------
    C_{ijkl} = \sum_{x=0}^{len(a_i)}{a_{ixkl} * b_{xjkl}}
    '''
    a_shape, b_shape = shape(a), shape(b)
    assert a_shape[-1] == b_shape[-2], "Y dimension of matrix A must match X dimension of matrix B"
    assert a_shape[:-2] == b_shape[:-2], "3rd and 4th dimensions of matrix A must match 3rd and 4th dimensions of matrix B"

    new_shape = (a_shape[0], a_shape[1], a_shape[2], b_shape[-1])
    zeros = zeros_from(new_shape)

    for l in range(new_shape[0]):
        for k in range(new_shape[1]):
            for i in range(new_shape[2]):
                for j in range(new_shape[3]):
                    zeros[l][k][i][j] = sum([a[l][k][i][x] * b[l][k][x][j] for x in range(len(a[l][k][i]))])
    
    return zeros

def zeros_from(shape: tuple) -> list:
    '''
    Returns a list of zeros with the given shape.
    '''
    zeros = [0] * math.prod(shape)
    return reshape_flattened_nd(zeros, shape)

def transpose(l: list) -> list:
    '''
    Transpose a 2d list.
    '''
    return [list(x) for x in zip(*l)]

def apply(l: list, f: callable) -> list:
    '''
    Apply a function to every element of an n-dimensional list.
    '''
    stored_shape = shape(l)
    applied = [f(x.data) if isinstance(x, Value) else Value(f(x)) for x in flatten_nd(l)]
    return reshape_flattened_nd(applied, stored_shape)

def tensor_sum(l: list) -> Value:
    '''
    Sum all elements of an n-dimensional list.
    '''
    return sum(flatten_nd(l))

def softmax(l: list) -> list:
    '''
    Get the softmax of an n-dimensional list.
    '''
    stored_shape = shape(l)
    exp = apply(l, math.exp)
    exp_sum = tensor_sum(exp)
    softmax = apply(exp, lambda x: x / exp_sum)
    return reshape_flattened_nd(softmax, stored_shape)

def add_tensors(l1: list, l2: list) -> list:
    '''
    Add two n-dimensional lists.
    '''
    stored_shape = shape(l1)
    added = [x + y for x, y in zip(flatten_nd(l1), flatten_nd(l2))]
    return reshape_flattened_nd(added, stored_shape)