import numpy as np

# Função sigmoide e sua derivada
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def derivada_sigmoide(x):
    return x * (1 - x)

# Inicialização de pesos aleatórios
np.random.seed(42)  # Para reprodutibilidade
input_neurons = 2   # Número de neurônios de entrada (A, B)
hidden_neurons = 2  # Número de neurônios na camada oculta
output_neurons = 1  # Número de neurônios de saída (previsão binária)

# Dados de entrada (A, B) e saída desejada
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
saidas_esperadas = np.array([[1], [0], [0], [1]])

# Inicializando pesos aleatórios para a camada oculta e de saída
pesos_entrada_oculta = np.random.rand(input_neurons, hidden_neurons)
pesos_oculta_saida = np.random.rand(hidden_neurons, output_neurons)

# Inicializando os vieses
viés_oculta = np.random.rand(1, hidden_neurons)
viés_saida = np.random.rand(1, output_neurons)

# Função para treinar a rede neural (retropropagação)
def treinar_mlp(entradas, saidas_esperadas, pesos_entrada_oculta, pesos_oculta_saida, viés_oculta, viés_saida, taxa_aprendizado, max_epocas=10000):
    for epoca in range(max_epocas):
        # Passo 1: Propagação para a frente (feedforward)
        soma_oculta = np.dot(entradas, pesos_entrada_oculta) + viés_oculta
        ativa_oculta = sigmoide(soma_oculta)

        soma_saida = np.dot(ativa_oculta, pesos_oculta_saida) + viés_saida
        saida_calculada = sigmoide(soma_saida)

        # Passo 2: Cálculo do erro
        erro = saidas_esperadas - saida_calculada
        erro_total = np.mean(np.abs(erro))  # Erro médio

        # Passo 3: Retropropagação
        derivada_saida = derivada_sigmoide(saida_calculada)
        gradiente_saida = erro * derivada_saida

        derivada_oculta = derivada_sigmoide(ativa_oculta)
        gradiente_oculta = gradiente_saida.dot(pesos_oculta_saida.T) * derivada_oculta

        # Passo 4: Atualização dos pesos e vieses
        pesos_oculta_saida += ativa_oculta.T.dot(gradiente_saida) * taxa_aprendizado
        pesos_entrada_oculta += entradas.T.dot(gradiente_oculta) * taxa_aprendizado
        viés_saida += np.sum(gradiente_saida, axis=0, keepdims=True) * taxa_aprendizado
        viés_oculta += np.sum(gradiente_oculta, axis=0, keepdims=True) * taxa_aprendizado

        # Passo 5: Impressão do progresso
        if epoca % 1000 == 0:
            acuracia = 100 - erro_total * 100
            print(f'Época: {epoca}, Erro: {erro_total:.4f}, Acurácia: {acuracia:.2f}%')

    return pesos_entrada_oculta, pesos_oculta_saida, viés_oculta, viés_saida

# Configurações de treinamento
taxa_aprendizado = 0.1
max_epocas = 10000

# Treinando a rede
pesos_entrada_oculta, pesos_oculta_saida, viés_oculta, viés_saida = treinar_mlp(
    entradas, saidas_esperadas, pesos_entrada_oculta, pesos_oculta_saida, viés_oculta, viés_saida, taxa_aprendizado, max_epocas
)

# Testando a rede após o treinamento
soma_oculta = np.dot(entradas, pesos_entrada_oculta) + viés_oculta
ativa_oculta = sigmoide(soma_oculta)
soma_saida = np.dot(ativa_oculta, pesos_oculta_saida) + viés_saida
saida_calculada = sigmoide(soma_saida)

# Exibindo as previsões
print("\nPrevisões após treinamento:")
for i in range(len(entradas)):
    print(f"Entrada: {entradas[i]} - Saída esperada: {saidas_esperadas[i]} - Saída calculada: {saida_calculada[i]}")

