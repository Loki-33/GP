import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('Qt5Agg')

def rbf(x1, x2, l=1.0, sigma_f=1.0):
	sqdist = np.subtract.outer(x1,x2)**2
	return sigma_f**2 * np.exp(-0.5/ l**2 * sqdist)


def linear(x1,x2,c=0.0):
	return np.outer(x1-c, x2-c)

def periodic(x1,x2, l=1.0, p=1.0, sigma_f=1.0):
	dists = np.pi * np.abs(np.subtract.outer(x1,x2))/p
	return sigma_f**2 * np.exp(-2 *(np.sin(dists)**2) / l**2)


def matern32(x1,x2,l=1.0, sigma_f=1.0):
	dists = np.abs(np.subtract.outer(x1,x2))
	sqrt3 = np.sqrt(3)
	return sigma_f**2 * (1+sqrt3 * dists / l) * np.exp(-sqrt3 * dists / l)


def sample_gp(kernel, x, n_samples=3):
	K = kernel(x,x)
	return np.random.multivariate_normal(np.zeros(len(x)), K + 1e-8*np.eye(len(x)), size=n_samples)

x = np.linspace(-5, 5, 100)

# List of kernels to compare
kernels = {
    "RBF Kernel": rbf,
    "Linear Kernel": linear,
    "Periodic Kernel": periodic,
    "Matérn 3/2 Kernel": matern32
}

# Plot samples
plt.figure(figsize=(16, 10))
for i, (name, kernel_fn) in enumerate(kernels.items(), 1):
    plt.subplot(2, 2, i)
    samples = sample_gp(kernel_fn, x, n_samples=5)
    for s in samples:
        plt.plot(x, s)
    plt.title(name)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)

plt.tight_layout()
plt.show()