import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parâmetros do modelo
N = 1_000_000       # População total
beta = 0.5          # Taxa de transmissão
sigma = 1/5.2       # Taxa de incubação (5,2 dias)
gamma = 1/14        # Taxa de recuperação (14 dias)

# Condições iniciais
I0 = 1              # Infectados iniciais
E0 = 0              # Expostos iniciais
R0 = 0              # Recuperados iniciais
S0 = N - I0 - E0 - R0  # Suscetíveis iniciais

# Intervalo de tempo (em dias)
t = np.linspace(0, 160, 160)

# Função com as equações diferenciais do modelo SEIR
def deriv(y, t, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Vetor de condições iniciais
y0 = S0, E0, I0, R0

# Integração numérica das equações
ret = odeint(deriv, y0, t, args=(N, beta, sigma, gamma))
S, E, I, R = ret.T

# Geração do gráfico
plt.figure(figsize=(10,6))
plt.plot(t, S, label='Suscetíveis', color='blue')
plt.plot(t, E, label='Expostos', color='orange')
plt.plot(t, I, label='Infectados', color='green')
plt.plot(t, R, label='Recuperados', color='red')
plt.xlabel('Dias')
plt.ylabel('Número de pessoas')
plt.title('Modelo SEIR para COVID-19')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
