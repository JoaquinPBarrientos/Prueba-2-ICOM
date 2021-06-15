# Prueba 2 simulación
import random
import matplotlib.pyplot as plt
import numpy as np

random.seed(198896900)

N = 10**5
bits = []
for i in range(N):
    bits.append(random.randint(0,1))

ook = np.zeros((N,8))

for i in range(len(bits)):
    if bits[i] == 0:
        ook[i] = np.array([0.0 for i in range(8)])
    else:
        ook[i] = np.array([1.0 for i in range(8)])

ook_vector = []

for i in range(len(ook)):
    for q in range(len(ook[i])):
        ook_vector.append(ook[i][q])

# plt.title("Señal modulada en OOK")
# plt.ylabel("Amplitud")
# plt.xlabel("Tiempo")
# plt.plot(range(len(ook_vector[:2000])),ook_vector[:2000])
# plt.show()


nrz = np.zeros((N,8))

for i in range(len(bits)):
    if bits[i] == 0:
        nrz[i] = np.array([-1.0 for i in range(8)])
    else:
        nrz[i] = np.array([1.0 for i in range(8)])

nrz_vector = []

for i in range(len(nrz)):
    for q in range(len(nrz[i])):
        nrz_vector.append(nrz[i][q])


plt.title("Señal modulada en NRZ")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo")
plt.axhline(y=0, color='k')
plt.plot(range(len(nrz_vector[:2000])),nrz_vector[:2000])
plt.show()

bppm = np.zeros((N,8))

for i in range(len(bits)):
    if bits[i] == 0:
        bppm[i] = np.array([1.0 if i <= 3 else 0.0 for i in range(8)])
    else:
         bppm[i] = np.array([0 if i <= 3 else 1 for i in range(8)])

bppm_vector = []

for i in range(len(bppm)):
    for q in range(len(bppm[i])):
        bppm_vector.append(bppm[i][q])


plt.title("Señal modulada en BPPM")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo")
plt.plot(range(len(bppm_vector[:700])),bppm_vector[:700])
plt.show()


Eb = 1
T = 1

A_ook = np.sqrt((2.0*Eb)/T)

for i in range(len(bits)):
    if bits[i] == 0:
        ook[i] = np.array([0.0 for i in range(8)])
    else:
        ook[i] = np.array([1.0*A_ook for i in range(8)])

ook_vector = []

for i in range(len(ook)):
    for q in range(len(ook[i])):
        ook_vector.append(ook[i][q])

plt.title("Señal modulada en OOK")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo")
plt.plot(range(len(ook_vector[:2000])),ook_vector[:2000])
plt.show()

A_nrz = np.sqrt((Eb)/T)

for i in range(len(bits)):
    if bits[i] == 0:
        nrz[i] = np.array([-1.0*A_nrz for i in range(8)])
    else:
        nrz[i] = np.array([1.0*A_nrz for i in range(8)])

nrz_vector = []

for i in range(len(nrz)):
    for q in range(len(nrz[i])):
        nrz_vector.append(nrz[i][q])


plt.title("Señal modulada en NRZ")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo")
plt.axhline(y=0, color='k')
plt.plot(range(len(nrz_vector[:2000])),nrz_vector[:2000])
plt.show()


A_bppm = np.sqrt(2*Eb)

for i in range(len(bits)):
    if bits[i] == 0:
        bppm[i] = np.array([1.0*A_bppm if i <= 3 else 0.0 for i in range(8)])
    else:
         bppm[i] = np.array([0.0 if i <= 3 else 1.0*A_bppm for i in range(8)])


bppm_vector = []

for i in range(len(bppm)):
    for q in range(len(bppm[i])):
        bppm_vector.append(bppm[i][q])

plt.title("Señal modulada en BPPM")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo")
plt.plot(range(len(bppm_vector[:700])),bppm_vector[:700])
plt.show()

EbNodb = [1.0,4.0,7.0,9.0,10.0]

sigma = []

for i in range(len(EbNodb)):
    EbNo = 10**((EbNodb[i]/10))
    sigma.append(np.sqrt(1/EbNo))

ruidos = np.zeros((5,8*N))
for i in range(len(sigma)):
    ruidos[i] = (np.random.normal(0,sigma[i],8*N))




ook_noise_vector = []
for i in  range(len(ook_vector)):
    ook_noise_vector.append(ook_vector[i] + ruidos[3][i])

plt.title("Señal modulada en OOK con ruido agregado")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo")
plt.axhline(y=0, color='k')
plt.step(range(len(ook_noise_vector[:80])),ook_noise_vector[:80])
plt.show()

nrz_noise_vector = []
for i in  range(len(nrz_vector)):
    nrz_noise_vector.append(nrz_vector[i] + ruidos[3][i])


plt.title("Señal modulada en NRZ con ruido agregado")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo")
plt.axhline(y=0, color='k')
plt.step(range(len(nrz_noise_vector[:80])),nrz_noise_vector[:80])
plt.show()


bppm_noise_vector = []
for i in  range(len(bppm_vector)):
    bppm_noise_vector.append(bppm_vector[i] + ruidos[3][i])

plt.title("Señal modulada en BPPM con ruido")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo")
plt.axhline(y=0, color='k')
plt.step(range(len(bppm_noise_vector[:80])),bppm_noise_vector[:80])
plt.show()


umbral_ook = (A_ook + 0)/2

s_ook=[]
for i in range(len(bits)):
    b = 0
    inter = ook_noise_vector[i*8:(i+1)*8]
    for q in range(len(inter)):
        b += inter[q]
    s_ook.append(b/8)

for i in range(len(s_ook)):
    if s_ook[i] > umbral_ook:
        s_ook[i] = 1
    else:
        s_ook[i] = 0

plt.subplot(2,1,1)
plt.title("Entrada original")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo")
plt.plot(range(len(bits[:100])),bits[:100])

plt.subplot(2,1,2)
plt.title("Salida de la señal")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo")
plt.plot(range(len(s_ook[:100])),s_ook[:100])

plt.show()

umbral_nrz = (A_nrz + -1*A_nrz)/2

s_nrz=[]

for i in range(len(bits)):
    b = 0
    inter = nrz_noise_vector[i*8:(i+1)*8]
    for q in range(len(inter)):
        b += inter[q]
    s_nrz.append(b/8)

for i in range(len(s_nrz)):
    if s_nrz[i] > umbral_nrz:
        s_nrz[i] = 1
    else:
        s_nrz[i] = 0

plt.subplot(2,1,1)
plt.title("Entrada original")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo")
plt.plot(range(len(bits[:100])),bits[:100])

plt.subplot(2,1,2)
plt.title("Salida de la señal")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo")
plt.plot(range(len(s_nrz[:100])),s_nrz[:100])

plt.show()


umbral_bppm = (A_bppm + 0)/2

s_bppm=[]
for i in range(len(bits)):
    b1 = 0.0
    b2 = 0.0
    inter = nrz_noise_vector[i*8:(i+1)*8]
    mitad1 = inter[0:3]
    mitad2 = inter[4:7]

    for i in range(len(mitad1)):
        b1 += mitad1[i]
        b2 += mitad2[i]
    b1 = b1/4
    b2 = b2/4

    
    if b1 > umbral_bppm:
        s_bppm.append(0.0) 
    elif b2 > umbral_bppm:
        s_bppm.append(1.0) 

plt.subplot(2,1,1)
plt.title("Entrada original")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo")
plt.plot(range(len(bits[:100])),bits[:100])

plt.subplot(2,1,2)
plt.title("Salida de la señal")
plt.ylabel("Amplitud")
plt.xlabel("Tiempo")
plt.plot(range(len(s_bppm[:100])),s_bppm[:100])

plt.show()

