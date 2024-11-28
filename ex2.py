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

if __name__ == "__main__":

    entrada = input("Digite uma sequência de números separados por espaço: ")


    try:
        sequencia = [float(num) for num in entrada.split()]
        if not sequencia:
            print("Você não digitou nenhum número. Tente novamente.")
        else:
            # Calcula os valores
            media = calcular_media(sequencia)
            variancia = calcular_variancia(sequencia, media)
            desvio_padrao = calcular_desvio_padrao(variancia)

            
            print(f"Média: {round(media, 2)}")
            print(f"Variância: {round(variancia, 2)}")
            print(f"Desvio Padrão: {round(desvio_padrao, 2)}")
    except ValueError:
        print("Entrada inválida. Certifique-se de digitar apenas números separados por espaço.")
