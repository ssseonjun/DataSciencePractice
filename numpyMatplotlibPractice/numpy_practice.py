import numpy as np
from numpy import random

wt=np.random.random(100)*50+40
ht=np.random.random(100)*60+140

bmi=wt/((ht*.01)**2)
bmi.resize((20,5))

for row in bmi:
    print(row)




# x=np.array([[1,2,3],[4,5,6],[7,8,9],
#             [11,12,13],[14,15,116],[17,18,19],
#             [21,22,23],[24,25,26],[27,28,29]])
# print(x[1,1])
# print(x[1,1:2])
# print(x[1:2,1:2])