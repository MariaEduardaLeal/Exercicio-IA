import random


def calcular_media(sequencia):
    soma = sum(sequencia)
    n = len(sequencia)
    media = soma / n
    return media

def calcular_variancia(sequencia, media):
    soma_quadrados = sum((x - media) ** 2 for x in sequencia)
    n = len(sequencia)
    variancia = soma_quadrados / n
    return variancia

def calcular_desvio_padrao(variancia):
    return variancia ** 0.5

sequencia = [random.randint(0, 1000) for _ in range(100)]

print((sequencia))

media = calcular_media(sequencia)

variancia = calcular_variancia(sequencia, media)

desvio_padrao = calcular_desvio_padrao(variancia)

print(f"Média: {media}")
print(f"Variância: {variancia}")
print(f"Desvio Padrão: {desvio_padrao}")
