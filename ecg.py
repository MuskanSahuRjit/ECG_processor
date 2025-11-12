import numpy as np
import matplotlib.pyplot as plt

# Simulate an ECG-like signal (sine wave with harmonics)
fs = 500                    # Sampling frequency in Hz
t = np.arange(0, 5, 1/fs)   # 5 seconds of data
ecg_clean = 1.5 * np.sin(2 * np.pi * 1 * t) + 0.5 * np.sin(2 * np.pi * 2 * t)

# Add 50Hz powerline noise
noise = 0.8 * np.sin(2 * np.pi * 50 * t)
ecg_noisy = ecg_clean + noise

# LMS filter parameters
N = 16              # Filter length
mu = 0.01           # Step size (learning rate)
w = np.zeros(N)     # Filter weights
x = np.zeros(N)     # Input buffer

y = []              # Filter output
e = []              # Error signal (cleaned ECG)

for n in range(len(t)):
    # Input vector x[n] (reference noise input, assume sine known)
    x[1:] = x[:-1]
    x[0] = np.sin(2 * np.pi * 50 * t[n])  # Reference 50Hz

    # Filter output y[n] = w^T * x
    y_n = np.dot(w, x)

    # Error e[n] = d[n] - y[n], where d[n] is the noisy ECG
    d_n = ecg_noisy[n]
    e_n = d_n - y_n

    # LMS weight update: w = w + μ * e * x
    w += mu * e_n * x

    # Store outputs
    y.append(y_n)
    e.append(e_n)

# Convert to numpy arrays
y = np.array(y)
e = np.array(e)

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(3,1,1)
plt.plot(t, ecg_noisy, label='Noisy ECG')
plt.legend()

plt.subplot(3,1,2)
plt.plot(t, e, label='Filtered ECG (LMS)', color='green')
plt.legend()

plt.subplot(3,1,3)
plt.plot(t, ecg_clean, label='Original Clean ECG', color='black')
plt.legend()

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# ... your signal generation, float_to_fixed(), fixed LMS loop, etc ...

# [1] Plot the results
plt.subplot(3,1,1)
plt.plot(t, label='Noisy ECG')
np.savetxt("noisy_ecg.txt")
plt.legend()

plt.subplot(3,1,2)
plt.plot(t, label='Filtered ECG (LMS)', color='green')
np.savetxt("filtered_lms.txt")
plt.legend()

plt.subplot(3,1,3)
plt.plot(t, label='Original Clean ECG', color='black')
np.savetxt("ref_signal.txt")
plt.legend()

plt.show()

# [2] Save data for FPGA testbench

print(noisy[:5])
print(ref[:5])
print(e_out[:5])


