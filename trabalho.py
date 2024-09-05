import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import pandas as pd

# Definindo os parâmetros do problema
L = 1.0
u_bar = 1.5
c0 = 1.5
t_values = np.linspace(0, 5, 100)  # Intervalo de tempo de 0 a 5 (100 pontos)
x_values = np.linspace(-10, 10, 500)  # Domínio espacial suficientemente grande

# Função para a solução da equação de advecção (Eq. 1)
def advection_solution(x, t):
    return np.where((x >= -L + u_bar * t) & (x <= L + u_bar * t), c0, 0)

# Função para a solução da equação de advecção-difusão (Eq. 9)
def advection_diffusion_solution(x, t, D):
    if t == 0:
        return np.where((x >= -L) & (x <= L), c0, 0)
    else:
        term1 = erf((L + (x - u_bar * t)) / np.sqrt(4 * D * t))
        term2 = erf((L - (x - u_bar * t)) / np.sqrt(4 * D * t))
        return 0.5 * c0 * (term1 + term2)

# Coeficientes de difusão
D_values = [1e-5, 1e-3, 1e-1]

# Seleção de tempos para análise detalhada
t_samples = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

# Seleção de valores de x mais relevantes para a tabela, próximos à região de interesse
# Aqui, escolhi pontos onde a frente de onda se desloca mais
x_sample_indices = [125, 200, 300, 375]  # Ajuste para capturar mais detalhes próximos à frente de onda

# Cálculo das soluções para diferentes instantes de tempo e difusões
solutions_adv = np.array([[advection_solution(x, t) for x in x_values] for t in t_values])
solutions_adv_diff = np.array([[[advection_diffusion_solution(x, t, D) for x in x_values] for t in t_values] for D in D_values])

# Criando DataFrames para armazenar os resultados
results_adv_df = pd.DataFrame()
results_adv_diff_df = pd.DataFrame()

# Adicionando os valores de x aos DataFrames
results_adv_df['x'] = x_values[x_sample_indices]
results_adv_diff_df['x'] = x_values[x_sample_indices]

# Mapeamento de tempos amostrados para os índices em t_values
t_indices = [np.argmin(np.abs(t_values - t)) for t in t_samples]

# Adicionando as soluções da equação de advecção ao DataFrame
for i, t_index in enumerate(t_indices):
    results_adv_df[f't={t_samples[i]}'] = np.round(solutions_adv[t_index, x_sample_indices], decimals=4)

# Adicionando as soluções da equação de advecção-difusão ao DataFrame
for idx, D in enumerate(D_values):
    for i, t_index in enumerate(t_indices):
        results_adv_diff_df[f'D={D} t={t_samples[i]}'] = np.round(solutions_adv_diff[idx][t_index, x_sample_indices], decimals=4)

# Salvando os resultados em arquivos CSV
results_adv_df.to_csv("resultados_adveccao.csv", index=False)
results_adv_diff_df.to_csv("resultados_adveccao_difusao.csv", index=False)

# Plotando os resultados e salvando os gráficos
plt.figure(figsize=(14, 8))

# Plotando a solução da equação de advecção para diferentes tempos
plt.subplot(2, 2, 1)
for i, t_index in enumerate(t_indices):
    plt.plot(x_values, solutions_adv[t_index], label=f't={t_samples[i]}')
plt.title('Solução da Equação de Advecção')
plt.xlabel('x')
plt.ylabel('c(x,t)')
plt.legend()
plt.savefig("grafico_adveccao.png")

# Plotando as soluções da equação de advecção-difusão para diferentes coeficientes de difusão
for idx, D in enumerate(D_values):
    plt.subplot(2, 2, idx + 2)
    for i, t_index in enumerate(t_indices):
        plt.plot(x_values, solutions_adv_diff[idx][t_index], label=f't={t_samples[i]}')
    plt.title(f'Solução da Equação de Advecção-Difusão (D={D})')
    plt.xlabel('x')
    plt.ylabel('c(x,t)')
    plt.legend()
    plt.savefig(f"grafico_adveccao_difusao_D={D}.png")

plt.tight_layout()
plt.show()
