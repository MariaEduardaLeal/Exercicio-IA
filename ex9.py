import numpy as np

# Dados de entrada e saída
entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Entradas A e B
saidas_esperadas = np.array([0, 1, 1, 1])  # Saídas esperadas

# Inicialização dos pesos e viés
pesos = np.zeros(2)  # Pesos para A e B
viés = 0  # Viés
taxa_aprendizado = 0.1  # Taxa de aprendizado

# Função de ativação (Degrau)
def funcao_ativacao(x):
    return 1 if x >= 0 else 0

# Função para treinar o perceptron
def treinar_perceptron(entradas, saidas_esperadas, pesos, viés, taxa_aprendizado, max_epocas=1000, acuracia_desejada=1.0):
    epoca = 0
    while epoca < max_epocas:
        erro_total = 0
        previsoes_corretas = 0
        for i in range(len(entradas)):
            # Cálculo da soma ponderada
            entrada_atual = np.dot(entradas[i], pesos) + viés
            # Cálculo da saída com a função de ativação
            saida_calculada = funcao_ativacao(entrada_atual)
            # Cálculo do erro
            erro = saidas_esperadas[i] - saida_calculada
            erro_total += abs(erro)
            if erro == 0:
                previsoes_corretas += 1
            # Atualização dos pesos e viés
            pesos += taxa_aprendizado * erro * entradas[i]
            viés += taxa_aprendizado * erro

            # Imprimir informações do treinamento a cada iteração
            print(f'Época {epoca + 1}, Exemplo {i + 1}:')
            print(f'  Entrada: {entradas[i]}, Saída esperada: {saidas_esperadas[i]}, Saída calculada: {saida_calculada}')
            print(f'  Pesos: {pesos}, Viés: {viés}, Erro: {erro}')
            print('---')

        acuracia = previsoes_corretas / len(entradas)
        print(f'Época {epoca + 1}: Erro total = {erro_total}, Acurácia = {acuracia * 100}%')

        # Verificar se a acurácia atingiu o valor desejado
        if acuracia >= acuracia_desejada:
            print("Rede neural alcançou a acurácia desejada!")
            break

        epoca += 1

    return pesos, viés

# Treinando o perceptron
pesos_finais, viés_final = treinar_perceptron(entradas, saidas_esperadas, pesos, viés, taxa_aprendizado)

# Mostrando os pesos finais e viés
print(f'Pesos finais: {pesos_finais}')
print(f'Viés final: {viés_final}')

# Testando o perceptron com as entradas
def prever(entradas, pesos, viés):
    previsoes = []
    for i in range(len(entradas)):
        entrada_atual = np.dot(entradas[i], pesos) + viés
        previsoes.append(funcao_ativacao(entrada_atual))
    return previsoes

# Realizando a previsão para as entradas
previsoes = prever(entradas, pesos_finais, viés_final)
print(f'Previsões: {previsoes}')
