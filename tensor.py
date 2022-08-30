'''
Apparently, all you need for a tensor is a list with a couple helper functions.
'''

# def is_valid_tensor(l: list) -> bool:
#     '''
#     Given an n-dimensional list, return True if it is a valid tensor.

#     A valid tensor is a n-dimentional list where the last dimention (n[-1]) is
#     the same for all elements.

#     This function recursively checks the dimensions of the list
#     '''
#     if not isinstance(l[0], list):
#         return True
#     dim = 0
#     return    

def shape(l: list) -> tuple:
    '''
    Recursively returns the shape of an n-dimensional list.
    '''
    if not isinstance(l[0], list):
        return (len(l),)
    return (len(l),) + shape(l[0])

def flatten_2d(l: list) -> list:
    '''
    Flatten a list of lists into a single list.
    '''
    return [item for sublist in l for item in sublist]

def view_by_head(l: list, d_k: int) -> list:
    '''
    Assume l is a matrix of shape (batch_size, sequence_length, d_model)

    Split the matrix into n_head matrices of shape (batch_size, sequence_length, n_heads, d_k)

    Return a transposed matrix of shape (batch_size, n_heads, sequence_length, d_k)
    '''
    matrix = []
    for seq in l:
        new_seq = []
        for token in seq:
            x = [token[x:x+d_k] for x in range(0, len(token), d_k)]
            new_seq.append(x)
        matrix.append(new_seq)
        
    return [list(token) for seq in matrix for token in zip(*seq)]

def reshape_flattened_2d(l: list, shape: tuple) -> list:
    '''
    Reshape a flattened list into a 2-dimensional list.
    '''
    return [l[i:i+shape[1]] for i in range(0, len(l), shape[1])]