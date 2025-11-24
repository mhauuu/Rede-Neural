# Importações das bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Estilo visual
sns.set(style="whitegrid")

print("--- SISTEMA DE PRECIFICAÇÃO IMOBILIÁRIA COM IA ---\n")

# =========================================================================================================
# 1. Dados 
dados_pernambuco = [
    {'Cidade': 'Recife', 'Area': 45, 'Quartos': 1, 'Banheiros': 1, 'Vagas': 1, 'Preco': 1800},
    {'Cidade': 'Recife', 'Area': 80, 'Quartos': 2, 'Banheiros': 2, 'Vagas': 1, 'Preco': 3200},
    {'Cidade': 'Recife', 'Area': 120, 'Quartos': 3, 'Banheiros': 3, 'Vagas': 2, 'Preco': 5500},
    {'Cidade': 'Recife', 'Area': 200, 'Quartos': 4, 'Banheiros': 4, 'Vagas': 3, 'Preco': 8500},
    {'Cidade': 'Recife', 'Area': 35, 'Quartos': 1, 'Banheiros': 1, 'Vagas': 0, 'Preco': 1200},
    {'Cidade': 'Jaboatao', 'Area': 50, 'Quartos': 2, 'Banheiros': 1, 'Vagas': 1, 'Preco': 1300},
    {'Cidade': 'Jaboatao', 'Area': 90, 'Quartos': 3, 'Banheiros': 2, 'Vagas': 1, 'Preco': 2800},
    {'Cidade': 'Jaboatao', 'Area': 130, 'Quartos': 3, 'Banheiros': 3, 'Vagas': 2, 'Preco': 3500},
    {'Cidade': 'Olinda', 'Area': 60, 'Quartos': 2, 'Banheiros': 1, 'Vagas': 1, 'Preco': 1100},
    {'Cidade': 'Olinda', 'Area': 100, 'Quartos': 3, 'Banheiros': 2, 'Vagas': 1, 'Preco': 2200},
    {'Cidade': 'Caruaru', 'Area': 50, 'Quartos': 2, 'Banheiros': 1, 'Vagas': 1, 'Preco': 900},
    {'Cidade': 'Caruaru', 'Area': 80, 'Quartos': 3, 'Banheiros': 2, 'Vagas': 1, 'Preco': 1500},
    {'Cidade': 'Petrolina', 'Area': 60, 'Quartos': 2, 'Banheiros': 1, 'Vagas': 1, 'Preco': 1000},
    {'Cidade': 'Petrolina', 'Area': 150, 'Quartos': 3, 'Banheiros': 3, 'Vagas': 2, 'Preco': 2800},
]
# Multiplicamento de dados
dados_pe = dados_pernambuco * 100
tabela = pd.DataFrame(dados_pe)

# Pré-procesamento desses dados
cidades_originais = tabela['Cidade']
tabela_numerica = pd.get_dummies(tabela, columns=['Cidade'], drop_first=True)
X = tabela_numerica.drop('Preco', axis=1)
y = tabela_numerica['Preco']
X_train, X_test, y_train, y_test, cid_train, cid_test = train_test_split(
    X, y, cidades_originais, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cérebro do Clóvis
modelo = MLPRegressor(hidden_layer_sizes=(400, 400, 400), max_iter=4000, random_state=100)

# ==============================================================================
# 2. Cáuculos de Treinamento

print("Treinando IA e gerando análises...")

# Learning Curve
train_sizes, train_scores, validation_scores = learning_curve(
    estimator=modelo, X=X_train_scaled, y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 5), cv=3, scoring='neg_mean_absolute_error', n_jobs=-1
)
train_errors = -train_scores.mean(axis=1)
validation_errors = -validation_scores.mean(axis=1)

# Treino Final
modelo.fit(X_train_scaled, y_train)
previsoes = modelo.predict(X_test_scaled)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='random')
X_test_tsne = tsne.fit_transform(X_test_scaled)

# ==============================================================================
# 3. Dashboard do projeto

fig, axs = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Relatório Técnico de IA - Mercado Imobiliário PE', fontsize=20, fontweight='bold')

# Gráfico 1: Learning Curve
axs[0, 0].plot(train_sizes, train_errors, 'o-', color="red", label="Erro de Treino")
axs[0, 0].plot(train_sizes, validation_errors, 'o-', color="green", label="Erro de Validação")
axs[0, 0].set_title('1. Learning Curve (Diagnóstico)', fontweight='bold')
axs[0, 0].set_ylabel('Erro (R$)')
axs[0, 0].set_xlabel('Volume de Dados')
axs[0, 0].legend()
axs[0, 0].grid(True, linestyle='--', alpha=0.5)

# Gráfico 2: Barras Real vs IA
df_visual = pd.DataFrame({'Real': y_test, 'IA': previsoes}).iloc[:10].reset_index(drop=True)
indices = np.arange(len(df_visual))
largura = 0.35
axs[0, 1].bar(indices - largura/2, df_visual['Real'], largura, label='Real', color='orange', alpha=0.6)
axs[0, 1].bar(indices + largura/2, df_visual['IA'], largura, label='IA', color='royalblue')
axs[0, 1].set_title('2. Teste de Precisão', fontweight='bold')
axs[0, 1].set_xticks(indices)
axs[0, 1].set_xticklabels([f'Casa {i+1}' for i in range(len(df_visual))])
axs[0, 1].legend()

# Gráfico 3: t-SNE
scatter = sns.scatterplot(
    x=X_test_tsne[:, 0], y=X_test_tsne[:, 1], hue=cid_test, 
    palette='bright', s=80, ax=axs[1, 0], legend='full'
)
axs[1, 0].set_title('3. Classificação t-SNE (Separação por Cidade)', fontweight='bold')
sns.move_legend(axs[1, 0], "upper left", bbox_to_anchor=(1, 1))

# Gráfico 4: Pizza
contagem = tabela['Cidade'].value_counts()
axs[1, 1].pie(contagem, labels=contagem.index, autopct='%1.0f%%', startangle=140, 
              colors=sns.color_palette("pastel"), pctdistance=0.85)
axs[1, 1].add_artist(plt.Circle((0,0),0.70,fc='white'))
axs[1, 1].set_title('4. Dados Utilizados', fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show(block=False) # block=False permite que o código de continuidade mesmo com a execução dos gráficos 

# ==================================================================================================================
# 4. Módulo de consultoria (Jorge)

print("\n" + "="*70)
print("RELATÓRIO DO CONSULTOR IMOBILIÁRIO (JORGE)")
print("="*70)
print(f"{'IMÓVEL':<10} | {'VALOR PEDIDO':<15} | {'JORGE CALCULOU':<15} | {'VEREDITO':<20}")
print("-" * 70)

# Aqui o robô (JORGE) vai pegar os 10 primeiros exemplos de imóveis para dar seu veredíto baseado no parâmetro dado a ele
for i in range(10):
    valor_real = y_test.iloc[i]
    valor_ia = previsoes[i]
    
    # Lógica do "Caro ou Barato"
    diferenca_percentual = (valor_real - valor_ia) / valor_ia
    
    if diferenca_percentual > 0.15: # Se 15% mais caro que a IA
        veredito = "CARO (Acima)"
    elif diferenca_percentual < -0.15: # Se 15% mais barato que a IA
        veredito = "BARATO (Oportunidade)"
    else:
        veredito = "PREÇO JUSTO"
        
    print(f"Casa {i+1:<5} | R$ {valor_real:<12.2f} | R$ {valor_ia:<12.2f} | {veredito}")

print("="*70)
print("Fim da execução. Feche o gráfico para sair.")
plt.show()