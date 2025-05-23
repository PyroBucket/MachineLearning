import numpy as np
import matplotlib.pyplot as plt

def evaluate(x):
    temperature = np.zeros_like(x[..., 0]) + 1500
    obj_temperature = np.zeros_like(x[..., 0])
    
    temperatures = [temperature]
    for t, v in enumerate(np.transpose(x)):
        delta_t = temperature
        delta_e = temperature - obj_temperature
        if 30 <= t <= 50:
            temperature = temperature - delta_t * 0.03
        else:
            temperature = temperature - delta_t * 0.01
        if t >= 40:
            temperature = temperature - delta_e * 0.025
            obj_temperature = obj_temperature + delta_e * 0.0125
        
        temperature = temperature + 50 * v
        temperatures.append(temperature)
    
    temperatures = np.array(temperatures).T
    cost = np.sum(x, axis=-1)
    diffs = np.sum(((temperatures[..., 40:] - 1500) / 500) ** 2, axis=-1)
    return -cost - diffs

def roulette_wheel_selection(population, fitness):
    fitness = fitness - fitness.min() + 1e-6
    probabilities = fitness / fitness.sum()
    indices = np.random.choice(len(population), size=len(population), p=probabilities)
    return population[indices]

def single_point_crossover(parent1, parent2, crossover_rate):
    if np.random.rand() < crossover_rate:
        point = np.random.randint(1, len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    return parent1, parent2

def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

# Parametry testowe
population_sizes = [50, 100, 200]
mutation_rates = [0.01, 0.05, 0.1]
crossover_rates = [0.6, 0.7, 0.9]
num_generations = 100
chromosome_length = 200

# Generowanie danych dla każdej kombinacji parametrów
results = {pop_size: [] for pop_size in population_sizes}

for pop_size in population_sizes:
    for mut_rate in mutation_rates:
        for cross_rate in crossover_rates:
            population = np.random.randint(0, 2, (pop_size, chromosome_length))
            best_fitness_per_gen = []

            # Algorytm genetyczny
            for generation in range(num_generations):
                fitness = np.array([evaluate(ind) for ind in population])
                best_fitness_per_gen.append(fitness.max())

                # Selekcja, krzyżowanie i mutacja
                selected_population = roulette_wheel_selection(population, fitness)
                new_population = []
                for i in range(0, pop_size, 2):
                    parent1 = selected_population[i]
                    parent2 = selected_population[i+1]
                    child1, child2 = single_point_crossover(parent1, parent2, cross_rate)
                    new_population.append(mutate(child1, mut_rate))
                    new_population.append(mutate(child2, mut_rate))
                
                population = np.array(new_population)

            results[pop_size].append(best_fitness_per_gen)

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle("Wpływ makroparametrów na algorytm genetyczny (dla różnych rozmiarów populacji)", fontsize=16)

for i, (pop_size, fitness_data) in enumerate(results.items()):
    ax = axes[i]
    ax.set_title(f"Populacja: {pop_size}")
    ax.set_xlabel("Generacja")
    if i == 0:
        ax.set_ylabel("Najlepsza przystosowanie")

    for j, mut in enumerate(mutation_rates):
        for k, cross in enumerate(crossover_rates):
            idx = j * len(crossover_rates) + k
            ax.plot(fitness_data[idx], label=f"Mut: {mut}, Cross: {cross}")

    ax.legend(loc="lower right", fontsize=8)

plt.show()
