import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

class EpidemicModel():
    def __init__(self, N, i0, g, b):
        
        # Politics:
            ## SIR no interventions
        self.i0 = i0
        self.N = N
        self.g = g
        self.b = b
        
        
        self.s = N - i0
        self.i = i0
        self.r = 0
            
        # Time
        self.t = 0
        self.dt = 1
        
        # history
        self.history = []
    
    def step(self):
        beta = self.b
        gamma = self.g
        
        # Flow
        ds = -beta * self.s * self.i
        di = (beta * self.s * self.i) - (gamma * self.i)
        dr = gamma * self.i
        
        # Update Stocks
        self.s+= ds*self.dt
        self.i+= di*self.dt
        self.r+= dr*self.dt
        
        # Time step
        self.t+=self.dt

        # Save
        self.history.append([self.t, self.s, self.i, self.r])
    
    def sim(self, time_limit=None, sim_limit = None):
        if time_limit!=None and time_limit>0:
            while self.t <= time_limit:
                self.step()

        if sim_limit!= None:
            for _ in range(sim_limit):
                self.step()

    
    def plot(self):
        history_dict = [{"t": h[0], "S": h[1], "I": h[2], "R": h[3]} for h in self.history]
        df = pd.DataFrame(history_dict)
        
        plt.title(f"Epidemia con Politicas")
        plt.plot(df["t"], df["S"], label="Susceptibles")
        plt.plot(df["t"], df["I"], label="Infectados")
        plt.plot(df["t"], df["R"], label="Recuperados")
        plt.xlabel("Dia")
        plt.ylabel("Poblacion")
        plt.tight_layout()
        plt.legend()
        plt.show()
        

model = EpidemicModel(
    N=10000,
    i0 = 10,
    b=0.35,
    g=0.1
)

model.sim(time_limit=500) # 100 dias
model.plot()
