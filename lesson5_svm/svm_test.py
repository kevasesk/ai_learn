import math, random, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

arr1 = np.array([
    [1, 1], 
    [2, 2]
])

operation = np.array([
    [1, 0], 
    [1, 1]
])

result = arr1.dot(operation)

print(result)


plt.subplot(131)
plt.title('arr1')
plt.plot(arr1[:, 0], arr1[:, 1], c='r', marker='o')
plt.axis('square')
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid()

plt.subplot(132)
plt.title('operation')
plt.plot(operation[:, 0], operation[:, 1], c='g', marker='o')
plt.axis('square')
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid()

plt.subplot(133)
plt.title('result')
plt.plot(result[:, 0], result[:, 1], c='b', marker='o')
plt.axis('square')
plt.xlim(-6, 6)
plt.ylim(-6, 6)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid()

plt.show()