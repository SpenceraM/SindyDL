import pandas as pd
import numpy as np
import pickle
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class DynamicSystem(ABC):
    def __init__(self, dt, t_end, f):
        self.x = None
        self.t = None
        self.dt = dt
        self.t_end = t_end
        self.coefficient_names = None
        self.state_names = None

    @abstractmethod
    def f(self, x, t):
        pass

    def step(self):
        self.x = self.x + self.dt * self.f(self.x, self.t)
        self.t = self.t + self.dt
        return self.x

    def run(self, x0, t0=0):
        self.x = x0
        self.t = t0
        xs = []
        while self.t < self.t_end:
            xs.append(self.step())
        return np.array(xs)

    def plot(self): # display in relevant space
        pass

    def get_derivative(self, states):  # save the trajectory and derivatives
       return np.gradient(states, axis=0, edge_order=2)/self.dt

    def print_coefficients(self, coefficients):
        if self.coefficient_names is None:
            self.get_library()
        print("Dynamic Equations:")
        for state_var in range(coefficients.shape[1]):
            eq_str = self.state_names[state_var] + " = "
            for lib_var in range(coefficients.shape[0]):
                if coefficients[lib_var,state_var] != 0:
                    eq_str += str(round(coefficients[lib_var,state_var],2)) + self.coefficient_names[lib_var] +  ' + '
                    # print(coefficients[i,j], self.coefficient_names[i], " = ", self.state_names[j])
            print(eq_str[:-2])


class Lorenz(DynamicSystem):
    def __init__(self, dt, t_end, f = None, sigma=np.nan, rho=np.nan, beta=np.nan):
        super().__init__(dt, t_end, f)
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.coefficient_names = ['1', 'x', 'y', 'z', 'x^2', 'xy', 'xz', 'y^2', 'yz', 'z^2']
        self.state_names = ['x', 'y', 'z']
        if f is not None:
            self.f = f

    def f(self, x, t):
        x1 = x[0]  # x1 = x
        x2 = x[1]  # x2 = y
        x3 = x[2]  # x3 = z
        return np.array([self.sigma * (x2 - x1), x1 * (self.rho - x3) - x2, x1 * x2 - self.beta * x3])

    @staticmethod
    def plot(xs):
        N = xs.shape[0]
        ax = plt.figure().add_subplot(projection='3d')
        ax.scatter(xs[:, 0], xs[:, 1], xs[:, 2],c=np.arange(N)/N, cmap='jet', marker = '.', s=.5)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.zaxis.set_ticklabels([])
        return ax

def estimated_f(x,t):
    x1 = x[0]  # x1 = x
    x2 = x[1]  # x2 = y
    x3 = x[2]  # x3 = z
    return np.array([9.65798187*x2*x3,-10.8217325*x1 - 10.9383841*x2, -6.84740 - 2.71854*x3 - 2.746*x1*x2 - .21334*x2**2])


if __name__ == '__main__':
    with open('experiment_results_202307121305.pkl', 'rb') as f:
        data = pickle.load(f)


    mask = np.array(data['sindy_mask'][0])
    sindy_coefficients = np.array(data['sindy_coefficients'][0])
    sindy_res = mask * sindy_coefficients

    lorenz = Lorenz(dt=0.0005, t_end=50, f=None, sigma=10, rho=28, beta=8 / 3)
    states = lorenz.run(np.array([1, 1, 1]), t0=0)
    lorenz.plot(states)
    plt.title('Original Lorenz System')

    # experiment 0
    not_lorenz = Lorenz(dt=0.0005, t_end=50, f=estimated_f)
    not_states = not_lorenz.run(np.array([1, 1, 1]), t0=0)
    not_lorenz.plot(not_states)
    plt.title('Estimated Lorenz System')

    plt.show()