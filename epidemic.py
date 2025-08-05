import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

class EpidemicModel():
    def __init__(self, N, i0, g, b):
        
        # Initial conditions
        self.N = N
        
        self.s0 = N- i0    
        self.i0 = i0
        self.r0 = 0
        
        self.v = 0
        
        self.case_threshold = i0+100
        # Constants
        self.g = g
        self.b = b
       
        # Initial SIR in vector form
        self.y0 = [self.s0, self.i0, self.r0]
        # Time
        self.t = None
        
        # Result
        self.result = []
    
    def get_v(self, I, S):
        if I >= self.case_threshold:
            self.v += 0.02
            self.case_threshold+=100
        Rt = (self.b/self.g) * (S/self.N)
        if Rt>1:
            self.v*=1.5
        return self.v
        
    def SIR(self, y, t):
        S, I, R = y
        v = self.get_v(I, S)
        beta, gamma = self.b, self.g
        
        ds = -(beta * S * I)/ self.N - (v*S)
        di = (beta*S*I)/self.N  - (gamma*I)
        dr = (gamma * I)/self.N  + (v*S)
        return [ds, di, dr]
        
    def sim(
        self,
        start,
        stop,
        num
    ):
        self.t = np.linspace(start, stop, num)
        self.result = odeint(self.SIR, self.y0, self.t)


    
    def plot(self):
        S, I, R = self.result.T
        plt.title(f"Epidemia con Politicas")
        plt.plot(self.t, S, label="Susceptibles")
        plt.plot(self.t, I, label="Infectados")
        plt.plot(self.t, R, label="Recuperados")
        plt.xlabel("Dia")
        plt.ylabel("Poblacion")
        plt.tight_layout()
        plt.legend()
        plt.show()
        

model = EpidemicModel(
    N=10000,
    i0=10,
    b=0.35,
    g=0.1
)

model.sim(0, 100, 500) # 100 dias
model.plot()
