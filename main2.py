import numpy as np
import random
import pygame
import time

def generate_random_cities(num_cities=20, coord_range=(-30, 30)):
    cities = {}
    for i in range(num_cities):
        city_name = chr(65 + i)
        x = random.uniform(*coord_range)
        y = random.uniform(*coord_range)
        cities[city_name] = (x, y)
    return cities

def distance(city1, city2):
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2)

def distance_total(travel):
    dist = 0
    for i in range(len(travel)):
        dist += distance(cities[travel[i]], cities[travel[(i + 1) % len(travel)]])
    return dist

def create_population(size, cities):
    population = []
    for _ in range(size):
        travel = list(cities.keys())
        travel.remove('A')
        random.shuffle(travel)
        travel = ['A'] + travel
        population.append(travel)
    return population

def evaluate_population(population):
    return [distance_total(travel) for travel in population]

def select(population, fitness, num_best):
    best_indices = np.argsort(fitness)[:num_best]
    return [population[i] for i in best_indices]

def crossover(travel1, travel2):
    start, end = sorted(random.sample(range(1, len(travel1)), 2))
    child = [None] * len(travel1)
    child[0] = 'A'
    child[start:end] = travel1[start:end]
    
    pointer = 0
    for city in travel2:
        if city not in child:
            while child[pointer] is not None:
                pointer += 1
            child[pointer] = city
    return child

def mutation(travel, mutation_rate):
    if random.random() < mutation_rate:
        i, j = random.sample(range(1, len(travel)), 2)
        travel[i], travel[j] = travel[j], travel[i]

def scale_coordinates(cities, width, height, margin=50):
    x_coords = [coord[0] for coord in cities.values()]
    y_coords = [coord[1] for coord in cities.values()]
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    scaled_cities = {}
    for city, (x, y) in cities.items():
        scaled_x = margin + (x - x_min) * (width - 2*margin) / (x_max - x_min)
        scaled_y = margin + (y - y_min) * (height - 2*margin) / (y_max - y_min)
        scaled_cities[city] = (scaled_x, scaled_y)
    
    return scaled_cities

def draw_route(screen, route, scaled_cities, color, font):
    # Dibujar líneas entre ciudades
    for i in range(len(route)):
        start = scaled_cities[route[i]]
        end = scaled_cities[route[(i + 1) % len(route)]]
        pygame.draw.line(screen, color, start, end, 2)
    
    # Dibujar ciudades y sus nombres
    for city, coord in scaled_cities.items():
        pygame.draw.circle(screen, (255, 0, 0), [int(coord[0]), int(coord[1])], 5)
        text = font.render(city, True, (0, 0, 0))
        screen.blit(text, (coord[0] + 10, coord[1] - 10))

def algorithm(cities, population_size=10, generations=10, mutation_rate=0.1):
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("TSP Genetic Algorithm Visualization")
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
    
    scaled_cities = scale_coordinates(cities, width, height)
    
    population = create_population(population_size, cities)
    best_travel = None
    best_distance = float('inf')
    worst_travel = None
    worst_distance = -float('inf')

    for generation in range(generations):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, None, None, None
        
        fitnesses = evaluate_population(population)
        best_idx = np.argmin(fitnesses)
        worst_idx = np.argmax(fitnesses)
        
        gen_best_travel = population[best_idx]
        gen_best_distance = fitnesses[best_idx]
        gen_worst_travel = population[worst_idx]
        gen_worst_distance = fitnesses[worst_idx]

        if gen_best_distance < best_distance:
            best_travel = gen_best_travel
            best_distance = gen_best_distance
        
        if gen_worst_distance > worst_distance:
            worst_travel = gen_worst_travel
            worst_distance = gen_worst_distance

        screen.fill((255, 255, 255))
        
        draw_route(screen, gen_best_travel, scaled_cities, (0, 255, 0), font)
        draw_route(screen, gen_worst_travel, scaled_cities, (255, 200, 200), font)
        
        
        gen_text = font.render(f"Generation: {generation + 1}", True, (0, 0, 0))
        best_text = font.render(f"Best Distance: {gen_best_distance:.2f}", True, (0, 255, 0))
        worst_text = font.render(f"Worst Distance: {gen_worst_distance:.2f}", True, (255, 0, 0))
        
        screen.blit(gen_text, (10, 10))
        screen.blit(best_text, (10, 30))
        screen.blit(worst_text, (10, 50))
        
        pygame.display.flip()
        clock.tick(0.5)  
        
        new_population = select(population, fitnesses, population_size // 2)
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(new_population, 2)
            child = crossover(parent1, parent2)
            mutation(child, mutation_rate)
            new_population.append(child)
        
        population = new_population

    screen.fill((255, 255, 255))
    
    draw_route(screen, best_travel, scaled_cities, (0, 255, 0), font)
    draw_route(screen, worst_travel, scaled_cities, (255, 200, 200), font)
    
    best_route_text = font.render("Best Route: " + " -> ".join(best_travel), True, (0, 0, 0))
    best_distance_text = font.render(f"Best Distance: {best_distance:.2f}", True, (0, 255, 0))
    worst_route_text = font.render("Worst Route: " + " -> ".join(worst_travel), True, (0, 0, 0))
    worst_distance_text = font.render(f"Worst Distance: {worst_distance:.2f}", True, (255, 0, 0))
    
    screen.blit(best_route_text, (10, 10))
    screen.blit(best_distance_text, (10, 30))
    screen.blit(worst_route_text, (10, 50))
    screen.blit(worst_distance_text, (10, 70))
    
    pygame.display.flip()
    
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
    
    pygame.quit()
    return best_travel, best_distance, worst_travel, worst_distance

cities = generate_random_cities(20, (-30, 30))
algorithm(
    cities, 
    generations=20, 
    population_size=10, 
    mutation_rate=0.05
)