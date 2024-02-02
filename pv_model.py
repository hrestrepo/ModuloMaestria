# Importar librerias
import numpy as np
import pandas as pd
from scipy.optimize import fsolve

class PVModel:
    """
    Clase para el panel fotovoltaico
    """
    def __init__(self, num_panels_series, num_panels_parallel):
        self.R_sh = 545.82
        self.k_i = 0.037
        self.T_n = 298
        self.q = 1.6021e-19
        self.n = 1.0
        self.K = 1.3806e-23
        self.E_g0 = 1.1
        self.R_s = 0.39
        self.num_panels_series = num_panels_series
        self.num_panels_parallel = num_panels_parallel
        self.I_sc = 9.35 * self.num_panels_parallel
        self.V_oc = 47.4 * self.num_panels_series
        self.N_s = 72

    def validate_inputs(self,G,T):
        if not isinstance(G, (int, float)) or G <= 0:
            raise ValueError('G debe ser un número positivo')
        if not isinstance(T, (int, float)) or T <= 0:
            raise ValueError('T debe ser un número positivo')
        if not isinstance(self.num_panels_series, int) or self.num_panels_series <= 0:
            raise ValueError('El número de paneles en serie debe ser un número positivo')
        if not isinstance(self.num_panels_parallel, int) or self.num_panels_parallel <= 0:
            raise ValueError('El número de paneles en paralelo debe ser un número positivo')

    def modelo_pv(self, G, T):
        self.validate_inputs(G,T)
        # I_rs corriente de saturación
        I_rs = self.I_sc / (np.exp((self.q*self.V_oc) / (self.n * self.N_s * self.K * T)) - 1)
        # I_o corriente de saturación inversa
        I_o = I_rs * (T/self.T_n) * np.exp((self.q * self.E_g0 * (1/self.T_n - 1 / T)) / (self.n *self.K))
        # I_ph Corriente fotogenerada
        I_ph = (self.I_sc + self.k_i * (T-298)) * (G/1000)

        # Crear un vector de voltajes de 0 hasta V_oc con 10000 puntos
        Vpv = np.linspace(0, self.V_oc, 10000)
        # Inicializar vectores de corriente y potencia
        Ipv = np.zeros_like(Vpv) # Inicializar vector de corriente
        Ppv = np.zeros_like(Vpv) # Inicializar vector de potencia

        # Funcion para resolver la ecuacion no lineal
        def f(I,V):
            return (I_ph - I_o * (np.exp((self.q * (V + I * self.R_s)) / (self.n * self.K * self.N_s * T)) - 1)-
                    (V + I * self.R_s) / self.R_sh - I)
        Ipv = fsolve(f, self.I_sc * np.ones_like(Vpv), args=(Vpv,)) # Resolver la ecuacion no lineal
        Ppv = Vpv * Ipv # Objetivo del modelo

        resultados = pd.DataFrame({'Corriente (A)': Ipv, 'Voltaje (V)': Vpv, 'Potencia (W)': Ppv})
        max_power_idx = resultados['Potencia (W)'].idxmax()
        Vmpp = resultados.loc[max_power_idx, 'Voltaje (V)']
        Impp = resultados.loc[max_power_idx, 'Corriente (A)']
        Pmax = resultados.loc[max_power_idx, 'Potencia (W)']
        return resultados, Vmpp, Impp, Pmax

def main():
    pv_model = PVModel(4, 3)
    G = 1000
    T = 25
    resultados, Vmpp, Impp, Pmax = pv_model.modelo_pv(G, T)
    print(resultados.head())
    print(f'Vmp = {Vmpp:.2f} V')
    print(f'Imp = {Impp:.2f} A')
    print(f'Pmax = {Pmax:.2f} W')

if __name__ == '__main__':
    main()


