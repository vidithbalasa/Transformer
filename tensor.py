'''
All you need for a tensor is a list with a couple helper functions.
'''

def is_valid_tensor(l: list) -> bool:
    '''
    Check if a list is a valid 2D tensor.

    1D tensor - Make sure every item is the same type.
    2D tensor - Make sure every item is the same type and has the same length.
    '''
    # 1D tensor
    if not isinstance(l[0], list):
        if not all(isinstance(item, type(l[0])) for item in l):
            return False
        return True

    # 2D tensor
    else:
        if not all(isinstance(item, list) for item in l):
            return False
        if not all(len(item) == len(l[0]) for item in l):
            return False
        return True

def flatten_2d(l: list) -> list:
    '''
    Flatten a list of lists into a single list.
    '''
    return [item for sublist in l for item in sublist]

def transpose_2d(l: list) -> list:
    '''
    Transpose a 2D list.
    '''
    return [list(i) for i in zip(*l)]

def transpose_1d(l: list) -> list:
    '''
    Transpose a 1D list.
    '''
    return [[i] for i in l]