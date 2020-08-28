import numpy as np
v = np.random.randn(10)
print('v',v)
maximum = np.max(v)
minimum = np.min(v)
print(maximum, minimum)

index_of_maximum = np.where(maximum)
index_of_minimum = np.where(minimum)

print(index_of_maximum[0][0])
print(index_of_minimum[0][0])