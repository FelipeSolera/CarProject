import pandas as pd
import numpy as np
from scipy import stats

# Atribua o nome do arquivo CSV à variável 'Car'
Car = "used_cars.csv"

# Lê o arquivo CSV em um DataFrame do Pandas
df = pd.read_csv(Car)

# Lista das colunas que deseja remover
colunas_a_remover = ["model", "model_year", "milage", "fuel_type", "engine", "transmission", "ext_col", "int_col", "accident", "clean_title"]
# Remove as colunas especificadas
df = df.drop(columns=colunas_a_remover)

# Mantém apenas as linhas que contenham 'Ford' ou 'Chevrolet' na coluna 'brand'
df = df[df['brand'].isin(['Ford', 'Chevrolet'])]

# Limpe a coluna de preços removendo cifrões e vírgulas, e converta para float
df['price'] = df['price'].str.replace('[\$,]', '', regex=True).astype(float)

# Divida o DataFrame em dois com base na marca (Ford e Chevrolet)
df_ford = df[df['brand'] == 'Ford']
df_chevrolet = df[df['brand'] == 'Chevrolet']

# Calcule a média e o desvio padrão para Ford e Chevrolet
mean_ford = np.mean(df_ford['price'])
mean_chevrolet = np.mean(df_chevrolet['price'])
std_ford = np.std(df_ford['price'])
std_chevrolet = np.std(df_chevrolet['price'])

# Calcule o tamanho das amostras
n_ford = len(df_ford)
n_chevrolet = len(df_chevrolet)

# Calcule a estatística t
t_statistic = (mean_ford - mean_chevrolet) / np.sqrt((std_ford**2 / n_ford) + (std_chevrolet**2 / n_chevrolet))

# Calcule os graus de liberdade
degrees_of_freedom = n_ford + n_chevrolet - 2

# Defina o nível de significância
alpha = 0.05

# Calcule o valor crítico do teste t para um teste de duas caudas
t_critical = stats.t.ppf(1 - alpha / 2, degrees_of_freedom)

# Calcule o p-valor usando a distribuição t de Student
p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), degrees_of_freedom))

# Imprima a estatística t, o valor crítico e o p-valor
print(f"Estatística t: {t_statistic}")
print(f"Valor Crítico (Two-Tail): {t_critical}")
print(f"P-valor: {p_value}")

# Verifique se a estatística t está fora do intervalo crítico
if np.abs(t_statistic) > t_critical:
    print("A diferença nas médias de preço entre Ford e Chevrolet é estatisticamente significativa.")
else:
    print("A diferença nas médias de preço entre Ford e Chevrolet não é estatisticamente significativa.")