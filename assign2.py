from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

def plot_signals(x, y, z, time):
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(time, x)
    plt.ylabel('5 Hz')

    plt.subplot(3, 1, 2)
    plt.plot(time, y)
    plt.ylabel('10 Hz')

    plt.subplot(3, 1, 3)
    plt.plot(time, z)
    plt.ylabel('20 Hz')
    plt.xlabel('Time [s]')
    plt.suptitle('x(t)')
    plt.show()

def generate_signal(limit, rms, T=1, dt=0.001, seed=42):
        np.random.seed(seed)

        # computed values
        limit_rad = limit*2*np.pi
        delta_omega = 2*np.pi/T
        # delta_omega = 1
        time = np.arange(0, T, dt)

        omegas = np.arange(0, limit_rad, delta_omega)
        N = len(omegas)

        x = np.random.randn(int(N))
        y = np.random.randn(int(N))

        a_0 = 1.2+0j
        cpx_p, cpx_n  = x+1j*y, x-1j*y

        coef = np.hstack((a_0, cpx_p[::-1], cpx_n))

        signal = np.fft.ifft(coef, n=len(time))  # this will pad with 0 for
                                                 # freqs outside the limit

        f_rms = lambda x: np.sqrt((x.real**2).sum()*dt)

        rms1 = f_rms(signal)
        # print 'Current RMS power:', rms1
        signal *= rms/rms1
        # print 'New RMS power:', f_rms(signal.real)
        print 'Sum of imaginary components in x(t):', signal.imag.sum()

        return signal, coef

## Part 1.1)
# a
x1, X1 = generate_signal(5, 0.5)
x2, X2 = generate_signal(10, 0.5)
x3, X3 = generate_signal(20, 0.5)

plot_signals(x1, x2, x3, np.arange(0, 1, 0.001))

# b
norms = []
for seed in range(100):
    x, X = generate_signal(10, 0.5, seed=seed)
    norms.append(map(np.linalg.norm, X))

norms = np.array(norms)
avg_norms = np.mean(norms, axis=0)

if 1:
    plt.figure()
    plt.plot(avg_norms[1:])
    plt.show()
