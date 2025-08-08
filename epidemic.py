import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint, solve_ivp

class EpidemicModel():
    def __init__(self, N, i0, g, b):
        # Initial conditions
        self.N = N
        self.s0 = N- i0    
        self.i0 = i0
        self.r0 = 0
        
        self.v = 0.00
        self.case_threshold = i0+100
        # Constants
        self.g = g
        self.b = b
        # Initial SIR in vector form
        self.y0 = [self.s0, self.i0, self.r0]
        # Time
        self.t = None
        # Result
        self.lotteries = []
        self.result = []
        
    def mandate_SIR(self,t, y):
        S, I, R = y
        beta, gamma = self.b, self.g
        ds = -(beta* S*I)/ self.N 
        di = (beta*S*I)/self.N - (gamma * I)
        dr = (gamma * I)
        return [ds, di, dr]

    def SIR(self, y, t):
        S, I, R = y
        # Lottery mode
        if (self.mode == 'lottery'):
            if I >= self.case_threshold:
                self.v += 0.02
                self.case_threshold+=100
                self.lotteries.append(t)
        
        # Mandate Mode
        v = 0 if self.mode=='normal' else self.v
        beta, gamma = self.b, self.g
        
        ds = -(beta * S * I)/ self.N - (v*S)
        di = (beta*S*I)/self.N  - (gamma*I)
        dr = (gamma * I) + (v*S)
        
        return [ds, di, dr]
        
    def sim(
        self,
        time,
        mode
    ):
        """
        time: tupla tipo (stop, num) utilizadas para definir el espacio linear
                del modelo sir, especificamente cuando terminar, y la cantidad de
                puntos intermedios, respectivamente
                
        mode: puede tomar los valores 'normal', 'lottery' o 'mandate' para las tres politicas
        """
        self.mode = mode
        if(mode=='mandate'):
            self.result = {}
            stop, num = time
            beta, gamma, N = self.b, self.g, self.N
            def rt_event(t, y):
                S = y[0]
                Rt = (beta/gamma) * (S/N)
                return min(t-25, Rt - 1)
            rt_event.terminal = True
            rt_event.direction = 1
            
            t_eval = np.linspace(0, stop, num)
            sol1 = solve_ivp(self.mandate_SIR, [0, stop], self.y0, events=rt_event,  dense_output=True, t_eval=t_eval)

            if sol1.status == 1:
                t_event = sol1.t_events[0][0]
                y_event = sol1.y_events[0][0]
                
                # Modify y_event instantly
                y_event[0] =  y_event[0] //2  # 50% reduction
                y_event[2] +=y_event[0]//2
                # Restart integration from event point with modified state
                t_eval = np.linspace(t_event, stop, num)
                sol2 = solve_ivp(self.mandate_SIR, [t_event, stop], y_event, dense_output=True, t_eval=t_eval)
                t_combined = np.hstack((sol1.t, sol2.t[1:]))
                y_combined = np.hstack((sol1.y, sol2.y[:, 1:]))
                
                self.result["t"] = t_combined
                self.result["y"] = y_combined
            else:
                self.result["t"]= sol1.t
                self.result["y"] = sol1.y
        else:
            stop, num = time
            self.t = np.linspace(0, stop, num)
            self.result = odeint(self.SIR, self.y0, self.t)
    
    def plot(self):
        if self.mode!='mandate':
            # odeint
            S, I, R= self.result.T
        else:
            # solve_ivp
            S, I, R = self.result["y"]
            self.t = self.result["t"]
            
        # Plot
        plt.figure(figsize=(10,5))
        if self.mode == 'normal':
            plt.title(f"Epidemia - Modelo SIR clasico\nβ={self.b}, γ={self.g}")
        if self.mode == 'lottery':
            plt.title(f"Epidemia - Loteria dv += 0.02 cada 100 casos\nβ={self.b}, γ={self.g}")
        if self.mode == 'mandate':
            plt.title(f"Epidemia - Refuerzo inmediato %50 con Rt > 1 y T>=25\nβ={self.b}, γ={self.g}")
        plt.plot(self.t, S/self.N, label="Susceptibles")
        plt.plot(self.t, I/self.N, label="Infectados")
        plt.plot(self.t, R/self.N, label="Recuperados")
        if len(self.lotteries) > 0:
            for i, l in enumerate(self.lotteries):
                if i == 0:
                    plt.axvline(l, linestyle=':', color='r', label='Lottery')
                else:
                    plt.axvline(l, linestyle=':', color='r')
        
        plt.xlabel("Dia")
        plt.ylabel("Poblacion")
        plt.tight_layout()
        plt.legend()
        plt.show()
     
def run_politics():
    # SIR no interventions
    m1 = EpidemicModel(
        N=10000,
        i0=10,
        b=0.35,
        g=0.1,
    )
    m1.sim((200, 500), mode='normal') # 100 dias
    m1.plot()
    
    # Lottery
    m2 = EpidemicModel(
        N=10000,
        i0=10,
        b=0.35,
        g=0.1,
    )
    m2.sim((200, 500), mode='lottery') # 100 dias
    m2.plot()
    
    # Mandate
    m3 = EpidemicModel(
        N=10000,
        i0=10,
        b=0.35,
        g=0.1,
    )
    m3.sim((200, 500), mode='mandate') # 100 dias
    m3.plot()
   
if __name__ == "__main__":
    run_politics()

