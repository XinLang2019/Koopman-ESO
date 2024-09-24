import numpy as np

class Estimator:
    def __init__(self, N, dt=0.01):
        
        self.N = N  # Estimated dimensions， in franka is 7
        
        self.alpha1 = np.ones(N)*5
        self.alpha2 = np.ones(N)*0.25
        self.delta = np.ones(N)*0.01
        
        self.beta01 = np.ones(N)*200
        self.beta02 = np.ones(N)*1000
        self.beta03 = np.ones(N)*2000
        
        self.dt = dt
        self.z1 = np.zeros(N)
        self.z2 = np.zeros(N)
        self.z3 = np.zeros(N)
        self.e = np.zeros(N)
        self.u = np.zeros(N)
        
    def smooth_sign(self, x):
        return np.sign(x)

    def fal(self, e, alpha):
        if np.abs(e) > self.delta:
            return np.power(np.abs(e), alpha) * self.smooth_sign(e)
        else:
            return e / np.power(self.delta, 1 - alpha)
    
    def get_u(self, u):
        self.u = u

    # estimate one of the system channals
    def sub_estimate_hat(self, x, i, u):
        self.e[i] = self.z1[i] - x[i]
        self.z1[i] = self.z1[i] + self.dt * (self.z2[i] - self.beta01[i] * self.e[i])
        self.z2[i] = self.z2[i] + self.dt * (self.z3[i] - self.beta02[i] * self.fal(self.e[i], self.alpha1[i]) + u[i])
        self.z3[i] = self.z3[i] - self.dt * self.beta03[i] * self.fal(self.e[i], self.alpha2[i])

    def estimate_hat(self, x):
        for i in range(self.N):
            self.sub_estimate_hat(x[i], self.u[i])
        