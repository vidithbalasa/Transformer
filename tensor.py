'''
Apparently all you need for a tensor is a list with a couple helper functions.
'''

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
    if len(og_shape) == 1:
        return matrix
    splits = len(matrix) // og_shape[0]
    return [reshape_flattened_nd(matrix[x:x+splits], og_shape[1:]) for x in range(0, len(matrix), splits)]

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
    
    return [list(map(list, zip(*seq))) for seq in matrix]