import numpy as np

from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from lmfit import Parameters, minimize

# Make some fake data 1-D
x = np.linspace(-10, 10, 1000)
mu_real = 10
mu_imag = -3

# Function to make fake data
def complex_gaussin(x, mu_real, mu_imag):
    return np.exp(-(x-mu_real)**2) + np.exp(-(x-mu_imag)**2) * 1j

def ft_gauss(x,mu):
	return np.exp(1j*mu*1/x)*np.exp(-(1/x)**2)

y = complex_gaussin(x, mu_real, mu_imag)
y = ft_gauss(x, mu_real)

# Plot to make sure it what we want
plt.figure(figsize=(12,6))
plt.plot(1/x, np.real(y), 'r', label='Real')
plt.plot(1/x, np.imag(y), 'b', label='Imag')

# When doing leastsquare minimisating need to define a cost function which in this case returns the difference
# between our model and data
def cost_function(par, x_obs, y_obs):
    """
    Cost function for 1-D complex gaussian
    """
    par = par.valuesdict()
    mu = par['mu']
    # y_model = complex_gaussin(x_obs, params[0], params[1])
    y_model = ft_gauss(x_obs, mu)
    # Add the two components in quadrature
    y_diff = (y_obs.real - y_model.real)**2 + (y_obs.imag - y_model.imag)**2 #y_obs - y_model
    return np.sqrt(y_diff)#.real**2 + y_diff.imag**2)


# Do the fit
# fit = least_squares(cost_function, [0, 0], args=(x, y))
# print('True real mu: {0}, imag mu {1} \n Fit real mu {2}, imag mu {3}'.format(mu_real, mu_imag, *fit.x))

par = Parameters()
par.add("mu",0, True)
fit = minimize(cost_function, par, args=(x,y))
fit_mu = fit.params.valuesdict()['mu']
print('True mu: {0}\n Fit mu {1}'.format(mu_real, fit_mu))

#Overplot fit results

# y_fit = complex_gaussin(x, fit.x[0], fit.x[1])
y_fit = ft_gauss(x, fit_mu)
plt.plot(1/x, np.real(y_fit), 'k--', label='Real Fit')
plt.plot(1/x, np.imag(y_fit), 'y--', label='Imag Fit')
plt.legend()
plt.show()