import numpy as np

a = np.array([[[1, 2, 100], [300, 4, 5]], [[6, 700, 8], [9, 10, 110]]])

b =  np.where(a > 99, 20, a)

print(a)
print(b)