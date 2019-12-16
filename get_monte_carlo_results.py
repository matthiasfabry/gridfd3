import numpy as np
import glob
import re
import matplotlib.pyplot as plt

folder = None
try:
    folder = sys.argv[1]
except IndexError:
    print('please give a folder!')

k1s = list()
k2s = list()
with open(folder + '/k1s', 'r') as rk1, open(folder + '/k2s', 'r') as rk2:
    k1lines = rk1.readlines()
    k2lines = rk2.readlines()
    for i in range(len(k1lines)):
        line = re.split('\n', k1lines[i])
        k1s.append(float(line[0]))
        line = re.split('\n', k2lines[i])
        k2s.append(float(line[0]))
k1s = np.array(k1s).flatten()
k2s = np.array(k2s).flatten()
# print results
print(np.average(k1s))
print(np.std(k1s)/np.sqrt(len(k1s)))
print(np.average(k2s))
print(np.std(k2s)/np.sqrt(len(k2s)))

plt.figure()
plt.hist(k1s, bins=int(max(k1s)-min(k1s)))
plt.xlabel(r'$k1 (km/s)$')
plt.ylabel(r'$N$')
plt.figure()
plt.hist(k2s, bins=int(max(k2s)-min(k2s)))
plt.xlabel(r'$k2 (km/s)$')
plt.ylabel(r'$N$')
plt.show()
