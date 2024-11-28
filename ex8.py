from collections import deque
import heapq
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Função para criar a animação da busca
def animar_busca(grid, explorados, caminho, inicio, objetivo, titulo="Animação de Busca"):
    linhas, colunas = len(grid), len(grid[0])
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.title(titulo)
    ax.set_xticks(range(colunas))
    ax.set_yticks(range(linhas))
    ax.grid(color='black', linewidth=1)
    ax.set_xlim(0, colunas)
    ax.set_ylim(0, linhas)
    ax.invert_yaxis()  # Inverter o eixo Y para alinhar com a grade

    # Inicializar o gráfico
    patches = []

    # Desenhar a grade inicial
    for x in range(linhas):
        for y in range(colunas):
            ax.add_patch(plt.Rectangle((y, x), 1, 1, color='white', ec='black'))

    # Marcar o início e o objetivo
    x_ini, y_ini = inicio
    x_obj, y_obj = objetivo
    ax.add_patch(plt.Rectangle((y_ini, x_ini), 1, 1, color='green', label='Início (S)'))
    ax.add_patch(plt.Rectangle((y_obj, x_obj), 1, 1, color='red', label='Objetivo (G)'))

    # Função para atualizar os frames
    def update(frame):
        for patch in patches:
            patch.remove()
        patches.clear()

        # Mostrar os nós explorados até o frame atual
        for (x, y) in explorados[:frame]:
            patch = plt.Rectangle((y, x), 1, 1, color='lightblue', alpha=0.6)
            ax.add_patch(patch)
            patches.append(patch)

        # Mostrar o caminho final no último frame
        if frame == len(explorados):
            for (x, y) in caminho:
                patch = plt.Rectangle((y, x), 1, 1, color='blue', alpha=0.8)
                ax.add_patch(patch)
                patches.append(patch)

    # Criar a animação
    ani = FuncAnimation(fig, update, frames=len(explorados) + 1, interval=300, repeat=False)
    plt.legend(loc='upper left')
    plt.show()

# Algoritmo de Busca em Largura (BFS)
def bfs_animado(grid, start, goal):
    queue = deque([[start]])
    visited = set()
    visited.add(start)
    explorados = []

    while queue:
        path = queue.popleft()
        x, y = path[-1]
        explorados.append((x, y))

        if (x, y) == goal:
            return explorados, path

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_pos = (x + dx, y + dy)
            if 0 <= next_pos[0] < len(grid) and 0 <= next_pos[1] < len(grid[0]) and next_pos not in visited:
                queue.append(path + [next_pos])
                visited.add(next_pos)
    return explorados, None

# Algoritmo de Busca em Profundidade (DFS)
def dfs_animado(grid, start, goal):
    stack = [[start]]
    visited = set()
    explorados = []

    while stack:
        path = stack.pop()
        x, y = path[-1]
        if (x, y) not in visited:
            visited.add((x, y))
            explorados.append((x, y))

            if (x, y) == goal:
                return explorados, path

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and (nx, ny) not in visited:
                    stack.append(path + [(nx, ny)])
    return explorados, None

# Algoritmo de Busca A* (A-star)
def a_star_animado(grid, start, goal):
    def heuristica(p1, p2):
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    heap = [(0 + heuristica(start, goal), 0, [start])]
    visited = set()
    explorados = []

    while heap:
        _, custo, path = heapq.heappop(heap)
        x, y = path[-1]
        if (x, y) not in visited:
            visited.add((x, y))
            explorados.append((x, y))

            if (x, y) == goal:
                return explorados, path

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and (nx, ny) not in visited:
                    novo_custo = custo + 1
                    heapq.heappush(heap, (novo_custo + heuristica((nx, ny), goal), novo_custo, path + [(nx, ny)]))
    return explorados, None

# Algoritmo de Busca Gulosa
def busca_gulosa_animado(grid, start, goal):
    def heuristica(p1, p2):
        return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    heap = [(heuristica(start, goal), [start])]
    visited = set()
    explorados = []

    while heap:
        _, path = heapq.heappop(heap)
        x, y = path[-1]
        if (x, y) not in visited:
            visited.add((x, y))
            explorados.append((x, y))

            if (x, y) == goal:
                return explorados, path

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and (nx, ny) not in visited:
                    heapq.heappush(heap, (heuristica((nx, ny), goal), path + [(nx, ny)]))
    return explorados, None

# Algoritmo de Busca Bidirecional
def busca_bidirecional_animado(grid, start, goal):
    fronteira_inicial = deque([[start]])  # Fronteira a partir do início
    fronteira_final = deque([[goal]])    # Fronteira a partir do objetivo
    visitados_inicial = {start}          # Conjunto de visitados a partir do início
    visitados_final = {goal}             # Conjunto de visitados a partir do objetivo
    explorados = []                      # Lista para animação

    # Função para obter os vizinhos válidos
    def vizinhos(x, y):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]):
                yield nx, ny

    # Dicionários para reconstruir os caminhos
    pais_inicial = {start: None}
    pais_final = {goal: None}

    while fronteira_inicial and fronteira_final:
        # Expansão a partir do início
        path_inicial = fronteira_inicial.popleft()
        x, y = path_inicial[-1]
        explorados.append((x, y))

        for nx, ny in vizinhos(x, y):
            if (nx, ny) not in visitados_inicial:
                visitados_inicial.add((nx, ny))
                fronteira_inicial.append(path_inicial + [(nx, ny)])
                pais_inicial[(nx, ny)] = (x, y)

                # Verifica se encontramos a outra busca
                if (nx, ny) in visitados_final:
                    # Reconstruir o caminho completo
                    caminho = reconstruir_caminho(pais_inicial, pais_final, (nx, ny))
                    return explorados, caminho

        # Expansão a partir do objetivo
        path_final = fronteira_final.popleft()
        x, y = path_final[-1]
        explorados.append((x, y))

        for nx, ny in vizinhos(x, y):
            if (nx, ny) not in visitados_final:
                visitados_final.add((nx, ny))
                fronteira_final.append(path_final + [(nx, ny)])
                pais_final[(nx, ny)] = (x, y)

                # Verifica se encontramos a outra busca
                if (nx, ny) in visitados_inicial:
                    # Reconstruir o caminho completo
                    caminho = reconstruir_caminho(pais_inicial, pais_final, (nx, ny))
                    return explorados, caminho

    return explorados, None

# Função para reconstruir o caminho entre os dois lados
def reconstruir_caminho(pais_inicial, pais_final, intersecao):
    caminho_inicial = []
    atual = intersecao
    while atual:
        caminho_inicial.append(atual)
        atual = pais_inicial[atual]
    caminho_inicial.reverse()

    caminho_final = []
    atual = intersecao
    while atual:
        atual = pais_final[atual]
        if atual:  # Evita adicionar a interseção novamente
            caminho_final.append(atual)

    return caminho_inicial + caminho_final



# Configuração da grade e pontos de início e destino
grid = [[0] * 5 for _ in range(5)]
start = (0, 0)
goal = (4, 4)


# BFS
explorados_bfs, caminho_bfs = bfs_animado(grid, start, goal)
animar_busca(grid, explorados_bfs, caminho_bfs, start, goal, "Busca em Largura (BFS)")

# DFS
explorados_dfs, caminho_dfs = dfs_animado(grid, start, goal)
animar_busca(grid, explorados_dfs, caminho_dfs, start, goal, "Busca em Profundidade (DFS)")

# A*
explorados_a_star, caminho_a_star = a_star_animado(grid, start, goal)
animar_busca(grid, explorados_a_star, caminho_a_star, start, goal, "Busca A*")

# Gulosa
explorados_gulosa, caminho_gulosa = busca_gulosa_animado(grid, start, goal)
animar_busca(grid, explorados_gulosa, caminho_gulosa, start, goal, "Busca Gulosa")

# Busca Bidirecional
explorados_bidir, caminho_bidir = busca_bidirecional_animado(grid, start, goal)
animar_busca(grid, explorados_bidir, caminho_bidir, start, goal, "Busca Bidirecional")
