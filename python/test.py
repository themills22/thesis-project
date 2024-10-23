import itertools as it
import numpy as np
import scaler

range = np.arange(4).reshape((2, 2))
print(np.average(range, axis=1))