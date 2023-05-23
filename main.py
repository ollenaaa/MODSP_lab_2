import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)


def fourier_coefficient(x, k):
    A_k = 0
    B_k = 0
    for n in range(N):
        A_k += x[n] * np.cos(2 * np.pi * k * n / N)
        B_k += x[n] * np.sin(2 * np.pi * k * n / N)
    c_k = complex(A_k, -B_k) / N
    add = N
    mul = N * 3

    return c_k, add, mul


def find_ck(x, N):
    oper_add = 0
    oper_mul = 0
    Ck = np.zeros(N, dtype=complex)
    for k in range(N):
        Ck[k], num_add, num_mul = fourier_coefficient(x, k)
        oper_add += num_add
        oper_mul += num_mul
        print(f"C_{k} = {Ck[k]}")
    print("Number of addition:", oper_add)
    print("Number of multiplication:", oper_mul)
    print("Number of operation:", oper_add + oper_mul)
    return Ck


N = 31
x = np.random.random(N)

start_time_dft = time.time()

C = find_ck(x, N)

end_time_dft = time.time()

print("Time for DFT: ", end_time_dft - start_time_dft)

amp_dft = abs(C)
phase_dft = np.angle(C)

plt.stem(amp_dft)

plt.xlabel('Frequency')
plt.ylabel('Amplitude')
plt.title('Amplitude spectrum')

plt.show()


plt.stem(phase_dft)

plt.xlabel('Frequency')
plt.ylabel('Phase')
plt.title('Phase graph')

plt.show()

