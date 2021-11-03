import numpy as np

X = 0  # Cartesian indices
Y = 1
Z = 2

E = 0  # Fields
H = 1

L = 0  # Lower
U = 1  # Upper


class Array:
    def __init__(self, array=np.empty((0,)), ini=(0,)):
        if type(array) != np.ndarray:
            raise TypeError("Array objects must be initialized with a numpy.ndarray object.")
        elif len(array.shape) != len(ini):
            raise ValueError("Array objects range must match ini length.")
        self.array = array
        self.ini = ini
    
    def set(self, array=np.empty((0,)), ini=(0,)):
        self.__init__(array, ini)
    
    def customKey(self, key):
        if type(key) == int:
            key_ = key-self.ini[0]
        elif type(key) == slice:
            start = key.start
            stop = key.stop
            if start is not None:
                start -= self.ini[0]
            if stop is not None:
                stop -= self.ini[0]
            key_ = slice(start, stop, key.step)
        elif type(key) == tuple:
            key_ = tuple()
            for i in range(len(key)):
                if type(i) == int:
                    key_ += (key[i]-self.ini[i],)
                elif type(i) == slice:
                    start = key[i].start
                    stop = key[i].stop
                    if start is not None:
                        start -= self.ini[i]
                    if stop is not None:
                        stop -= self.ini[i]
                    key_ += (slice(start, stop, key[i].step),)
        else:
            raise TypeError("Array object must be indexed with integer, tuples or slices.")
        return key_
        
    def __getitem__(self, key):
        return self.array[self.customKey(key)]
    
    def __setitem__(self, key, item):
        self.array[self.customKey(key)] = item
