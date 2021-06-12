# Prueba 2 simulación
import random 
import matplotlib.pyplot as plt

random.seed(198896900)

N = 10**5
bits = []

for i in range(N):
    bits.append(random.randint(0,1))



plt.plot(range(N),bits)
plt.show()
print(range(10))