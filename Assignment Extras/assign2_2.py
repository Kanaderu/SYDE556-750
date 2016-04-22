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


def generate_signal(limit, rms, T=1, dt=0.001, seed=42, std=1):
    rad = lambda x: x*2*np.pi
    np.random.seed(seed)

    # computed values
    delta_omega = 2*np.pi/T
    time = np.arange(0, T, dt)
    N = int(0.5*T/dt)

    omegas = np.arange(0, delta_omega*N, delta_omega)

    if isinstance(std, int): 
        x = std*np.random.randn(N)
        y = std*np.random.randn(N)

        over_limit = np.where(omegas > rad(limit))[0]
        x[over_limit], y[over_limit] = 0, 0
    else:
        # print 'Picking Gaussian PSN'
        x = std(omegas, rad(limit))*np.random.randn(N)
        y = std(omegas, rad(limit))*np.random.randn(N)
        
        
    a_0 = 0+0j  # zero-term freq
    cpx_p, cpx_n  = x+1j*y, x-1j*y

    # negative flipped for ifft, from docs:
    # "order of decreasingly negative frequency"
    coef = np.hstack((a_0, cpx_p, cpx_n[::-1]))

    signal = np.fft.ifft(coef)

    f_rms = lambda x: np.sqrt((x.real**2).sum()*dt)

    rms1 = f_rms(signal)
    signal *= rms/rms1
    np.set_printoptions(precision=10)
    print signal
    print 'Sum of imaginary components in x(t):', np.abs(signal.imag).sum()

    return signal, coef


def plot_coefficients(time, std=1):
    omegas = np.fft.fftfreq(1001, 1/1001.)
    shifted_o = np.array(np.fft.fftshift(omegas), dtype=np.int)

    norms = []
    for seed in range(100):
        x, X = generate_signal(10, 0.5, seed=seed, std=std)
        norms.append(map(np.linalg.norm, X))

    norms = np.array(norms)
    avg_norms = np.mean(norms, axis=0)

    plt.figure()
    plt.plot(shifted_o, avg_norms[shifted_o])
    plt.xlabel('$\omega$', fontsize=10)
    plt.ylabel('| $X(\omega)$ |')
    plt.show()


def part_1_1():
    dt = 0.001
    T = 1
    time = np.arange(0, T+dt, dt)

    # a
    x1, X1 = generate_signal(5, 0.5)
    x2, X2 = generate_signal(10, 0.5)
    x3, X3 = generate_signal(20, 0.5)
    plot_signals(x1, x2, x3, time)

    # b
    plot_coefficients(time)

def part_1_2():
    exp_std = lambda w, b: np.exp(-w**2/(2*(b**2)))
    x1, X1 = generate_signal(5, 0.5, std=exp_std)
    x2, X2 = generate_signal(10, 0.5, std=exp_std)
    x3, X3 = generate_signal(20, 0.5, std=exp_std)

    plot_signals(x1, x2, x3, time)
    plot_coefficients(time, exp_std)

part_1_1()

## Part 1.2)
# a

