from micrograd.nn import Layer
from tensor import flatten_2d, shape, reshape_flattened_2d

class Linear(Layer):
    '''
    A single dense layer.
    '''
    def __init__(self, n_in, n_out, **kwargs):
        super().__init__(n_in, n_out, **kwargs)
        self.n_in = n_in
        self.n_out = n_out
    
    def __call__(self, x:list):
        '''
        Forward pass.
        '''
        # save dimentions and flatten if 2D
        if isinstance(x[0], list):
            self.dims = shape(x)
            x = flatten_2d(x)
        else:
            self.dims = None

        x = super().__call__(x)

        # reshape if 2D
        if self.dims and self.n_in == self.n_out:
            x = reshape_flattened_2d(x, self.dims)
        
        return x

    
    def __repr__(self):
        return f"Linear(n_in={self.n_in}, n_out={self.n_out})"