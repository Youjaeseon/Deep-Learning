import numpy as np

A = np.array([1, 2, 3, 4])

#배열이 몇 차원인지.
print(A.ndim)   # 1
#각 차원의 크기.
print(A.shape)  # (4,)

B = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print(B.ndim)   # 2
print(B.shape)  # (2, 3)