import pygame
import random
import numpy as np

# Inicializa o Pygame
pygame.init()

# Dimensões da tela
WIDTH, HEIGHT = 500, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Algoritmo Genético Camuflagem")

# Definições
NUM_BOLINHAS = 100
RAIO = 5
VELOCIDADE = 5
GERACOES = 1000

POPULACAO = 200

TARGET_RGB = np.array([50, 255, 130])
MUTATION_RATE = 0.1
ELITE_RATIO = 0.95

# Cores
BRANCO = (255, 255, 255)

# Função para gerar uma posição aleatória dentro da tela
def gerar_posicao_aleatoria():
    x = random.randint(RAIO, WIDTH - RAIO)
    y = random.randint(RAIO, HEIGHT - RAIO)
    return [x, y]

# Vetor para armazenar as posições das bolinhas
posicoes_bolinhas = [gerar_posicao_aleatoria() for _ in range(NUM_BOLINHAS)]


# Função para mover a bolinha aleatoriamente, respeitando os limites da tela
def mover_bolinha(posicao):
    direcao = random.choice(['cima', 'baixo', 'esquerda', 'direita'])
    
    if direcao == 'cima' and posicao[1] - VELOCIDADE > RAIO:
        posicao[1] -= VELOCIDADE
    elif direcao == 'baixo' and posicao[1] + VELOCIDADE < HEIGHT - RAIO:
        posicao[1] += VELOCIDADE
    elif direcao == 'esquerda' and posicao[0] - VELOCIDADE > RAIO:
        posicao[0] -= VELOCIDADE
    elif direcao == 'direita' and posicao[0] + VELOCIDADE < WIDTH - RAIO:
        posicao[0] += VELOCIDADE


def calc_fitness(chromosome, target_rgb):
    """
    Calculate the fitness of a given chromosome.

    Parameters:
        chromosome (np.ndarray): An array representing the chromosome.
        target_rgb (np.ndarray): The target RGB values.

    Returns:
        float: The mean absolute difference between the chromosome and target RGB values.
    """
    return np.mean(np.abs(chromosome - target_rgb))        

def evolve_population(population, target_rgb, mutation_rate, elite_ratio):
    """
    Evolve a population of chromosomes.

    Parameters:
        population (List[np.ndarray]): A list of arrays representing the chromosomes in the population.
        target_rgb (np.ndarray): The target RGB values.
        mutation_rate (float): The probability of a chromosome mutating.
        elite_ratio (float): The ratio of chromosomes to be selected as elites.

    Returns:
        List[np.ndarray]: The evolved population of chromosomes.
    """
    # Calculate fitness values for each chromosome in the population
    fitness_values = [calc_fitness(c, target_rgb) for c in population]

    # Select the chromosomes with the best fitness values as elites
    elite_index = np.argsort(fitness_values)[:int(len(population) * elite_ratio)]
    elites = [population[i] for i in elite_index]

    # Create a new population by mating the elites
    new_population = []
    while len(new_population) < len(population) - len(elites):
        # Select two parents randomly from the elites
        parent1, parent2 = random.choices(elites, k=2)

        # Create a child chromosome by combining the genes of the two parents
        child = np.zeros(parent1.shape)
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]

            # Mutate the child chromosome with a probability determined by mutation_rate
            if random.random() < mutation_rate:
                child[i] = np.random.randint(0, 256)

        # Add the child chromosome to the new population
        new_population.append(child)

    # Return the evolved population
    return elites + new_population


def simulate(size, target_rgb, generations, mutation_rate, elite_ratio):
    """
    Simulate the evolution of a population of chromosomes.

    Parameters:
        size (int): The size of the population.
        target_rgb (np.ndarray): The target RGB values.
        generations (int): The number of generations to evolve the population.
        mutation_rate (float): The probability of a chromosome mutating.
        elite_ratio (float): The ratio of chromosomes to be selected as elites.

    Returns:
        np.ndarray: The best chromosome found.
    """
    # Initialize the population with random RGB values
    population = [np.random.randint(0, 256, size=(3,)) for _ in range(size)]

    # Evolve the population over the specified number of generations
    for i in range(generations):
        # Evolve the population by selecting elites, mutating, and recombining
        population = evolve_population(population, target_rgb, mutation_rate, elite_ratio)

        # Get the individual with the best fitness (closest to target RGB)
        best_fit = min(population, key=lambda c: calc_fitness(c, target_rgb))

        # Print the generation number and the RGB values and fitness of the best individual
        print(f'Generation {i}: Best fitness {calc_fitness(best_fit, target_rgb)}, RGB values {best_fit}')

    # Return the individual with the minimum fitness value
    return min(population, key=lambda c: calc_fitness(c, target_rgb))

# Loop principal
def main():
    clock = pygame.time.Clock()
    estado = 0
    running = True

     # Initialize the population with random RGB values
    population = [np.random.randint(0, 256, size=(3,)) for _ in range(POPULACAO)]

    while running and estado < GERACOES:
        # Processa os eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Limpa a tela
        screen.fill(BRANCO)
        
        # Move e desenha cada bolinha
        for idx, posicao in enumerate(posicoes_bolinhas):
            mover_bolinha(posicao)
            pygame.draw.circle(screen, tuple(population[idx]), posicao, RAIO)


        population = evolve_population(population, TARGET_RGB, MUTATION_RATE, ELITE_RATIO)

        # Get the individual with the best fitness (closest to target RGB)
        best_fit = min(population, key=lambda c: calc_fitness(c, TARGET_RGB))

        # Print the generation number and the RGB values and fitness of the best individual
        print(f'Generation {estado}: Best fitness {calc_fitness(best_fit, TARGET_RGB)}, RGB values {best_fit}')    
        
        # Atualiza o display
        pygame.display.flip()
        
        # Controle de tempo (5 FPS)
        clock.tick(5)
        
        # Incrementa o estado
        estado += 1

    pygame.quit()

if __name__ == "__main__":
    main()
