import numpy as np
import random
import matplotlib.pyplot as plt

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

def algorithm(cities, population_size=10, generations=10, mutation_rate=0.1):
    population = create_population(population_size, cities)
    best_travel = None
    best_distance = float('inf')

    for generation in range(generations):
        fitnesses = evaluate_population(population)
        best_gen_travel = population[np.argmin(fitnesses)]
        best_gen_distance = min(fitnesses)

        if best_gen_distance < best_distance:
            best_travel = best_gen_travel
            best_distance = best_gen_distance

        new_population = select(population, fitnesses, population_size // 2)
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(new_population, 2)
            child = crossover(parent1, parent2)
            mutation(child, mutation_rate)
            new_population.append(child)

        population = new_population

        print(f"Generation {generation + 1}:")
        for i, travel in enumerate(population):
            travel_distance = distance_total(travel)
            print(f"  Route {i + 1}: {travel} -> Distance: {travel_distance:.2f}")
        
        print(f"Best route this generation: {best_gen_travel} -> Distance: {best_gen_distance:.2f}")
        print("-" * 50)

    return best_travel, best_distance

def plot_route(route, distance):
    route_coords = [cities[city] for city in route]
    route_coords.append(route_coords[0])
    x, y = zip(*route_coords)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='--', color='g')
    plt.title('Best Route', fontsize=15)
    plt.suptitle(f'Distance: {distance:.2f} - Route: {route}', fontsize=10)

    for city, coord in cities.items():
        plt.annotate(city, coord, textcoords="offset points", xytext=(0, 10), ha='center')

    plt.show()

cities = generate_random_cities(20, (-30, 30))

best_travel, best_distance = algorithm(cities, generations=10, population_size=10, mutation_rate=0.1)

print("Best overall route:", best_travel)
print(f'"Best overall distance:"{best_distance:.2f}')

plot_route(best_travel, best_distance)