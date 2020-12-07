import numpy as np
import sys

ff = 'labels.npy'

if len(sys.argv) > 1:
    ff = sys.argv[1]

lb = np.load(ff)
for i in range(len(lb)):
    if lb[i][0][1] < lb[i][1][1]:
        # check that the y value of the first point is
        # always less than the second
        continue
    print('changed y at index ', str(i))
    temp = lb[i][0][1].copy()
    lb[i][0][1] = lb[i][1][1].copy()
    lb[i][1][1] = temp.copy()

for i in range(len(lb)):
    if lb[i][0][0] < lb[i][1][0]:
        # check that the x value of the first point is
        # always less than the second
        continue
    print('changed x at index ', str(i))
    temp = lb[i][0][0].copy()
    lb[i][0][0] = lb[i][1][0].copy()
    lb[i][1][0] = temp.copy()

np.save(ff, lb, True)