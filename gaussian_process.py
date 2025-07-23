import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib 
matplotlib.use('Qt5Agg') #this tells matplotlub to use PyQt as the output window

def rbf(x1,x2,sigma_f=1.0, l=1.0):
	dist = np.subtract.outer(x1,x2)**2
	return sigma_f**2 * np.exp(-dist/(2*l**2))


def zero_mean(x):
	return np.zeros_like(x)

def linear_mean(x):
	return 0.5* x + 1.0


def gp(x_train, y_train, x_test, kernel, mean_func, noise= 1e-8):
	K = kernel(x_train, x_train) + noise * np.eye(len(X_train))
	K_s = kernel(x_train, x_test)
	K_ss =kernel(x_test, x_test) + noise * np.eye(len(x_test))

	m_train = mean_func(x_train)
	m_test = mean_func(x_test)

	K_inv = np.linalg.inv(K)
	mu_s = m_test + K_s.T @ K_inv @ (y_train - m_train)
	cov_s = K_ss - K_s.T @ K_inv @ K_s

	return mu_s, cov_s

X_train = np.array([-2.0, -1.2, -0.5, 0.3, 1.1, 1.8, 2.5])
y_train = np.array([0.5, -0.8, -0.3, 0.7, 0.9, -0.2, -1.1])


X_test = np.linspace(-3, 3, 100)


#zero mean
mu_zero, cov_zero = gp(X_train, y_train, X_test, rbf, zero_mean)
std_zero = np.sqrt(np.clip(np.diag(cov_zero), 0, np.inf))
print(std_zero)
print('---')

#linear mean
mu_linear, cov_linear = gp(X_train, y_train, X_test, rbf, linear_mean)
std_linear = np.sqrt(np.clip(np.diag(cov_linear), 0, np.inf))
print(std_linear)


#zero
plt.subplot(1, 2, 1)
plt.title("GP with Zero Mean")
plt.fill_between(X_test, mu_zero - 2*std_zero, mu_zero + 2*std_zero, alpha=0.2, label='Confidence Interval')
plt.plot(X_test, mu_zero, label='Mean Prediction')
plt.plot(X_train, y_train, 'rx', label='Training Points')
plt.legend()

#linear
plt.subplot(1, 2, 2)
plt.title("GP with Linear Mean (0.5x + 1)")
plt.fill_between(X_test, mu_linear - 2*std_linear, mu_linear + 2*std_linear, alpha=0.2, label='Confidence Interval')
plt.plot(X_test, mu_linear, label='Mean Prediction')
plt.plot(X_train, y_train, 'rx', label='Training Points')
plt.legend()

plt.tight_layout()
plt.show()
