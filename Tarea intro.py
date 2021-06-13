# Prueba 2 simulación
import random
import matplotlib.pyplot as plt
import numpy as np

random.seed(198896900)

N = 10**5
bits = []
for i in range(N):
    bits.append(random.randint(0,1))

bits = np.array(bits)

ook = np.zeros((N,8))

for i in range(len(bits)):
    if bits[i] == 0:
        ook[i] = np.array([0 for i in range(8)])
    else:
        ook[i] = np.array([1 for i in range(8)])
ook