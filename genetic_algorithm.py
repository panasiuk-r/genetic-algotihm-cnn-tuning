from threading import Thread
import random

from .cnn_individual import CNNIndividual
from .padding_mapping import padding_mapping
from .activation_mapping import activation_mapping

class GeneticAlgorithm:
    def __init__(self, population_size):
        self.population_size = population_size
        self.population = self.initialize_population()
        self.generation = 0
        self.best_individual_per_generation = []
        self.fitness_data = {
            'fitness_by_layer': [],
            'fitness_by_filter_num': [],
            'fitness_by_filter_size': [],
            'fitness_by_stride': [],
            'fitness_by_pooling_size': [],
            'fitness_by_dropout_prob': [],
            'fitness_by_beta_1': [],
            'fitness_by_beta_2': [],
            'fitness_by_learning_rate': [],
            'fitness_by_padding': [],
            'fitness_by_activation': []
        }

    
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            num_conv_layers = random.randint(1, 2)
            num_filters = random.randint(16, 64)
            filter_size = random.choice([3, 5])
            activation = random.randint(0, len(activation_mapping) - 1)
            pooling_size = random.choice([2, 3])
            dropout_prob =  random.uniform(-1, 1)
            beta_1 = random.uniform(0.8, 0.999)
            beta_2 = 1 - 10 ** (-random.uniform(1, 5))
            if dropout_prob < 0:
                dropout_prob = 0
            stride = random.choice([1, 2])
            padding = random.randint(0, len(padding_mapping) - 1)
            learning_rate = 10 ** (-random.uniform(1, 5))

            individual = CNNIndividual(
                num_conv_layers, num_filters, filter_size,activation, pooling_size, dropout_prob,
                stride, padding, learning_rate, beta_1, beta_2,
            )
            print(individual.__repr__())
            population.append(individual)
        
        return population
    
    # tournament selection
    def selection(self, tournament_size=3, p=0.75):
        selected_parents = []
        for _ in range(self.population_size):
            tournament = random.sample(self.population, tournament_size)
            tournament.sort(key=lambda ind: ind.fitness, reverse=True)
            
            for i in range(tournament_size):
                if random.random() < p * ((1 - p) ** i):
                    selected_parents.append(tournament[i])
                    break
            else:
                selected_parents.append(tournament[0])
        
        return selected_parents

    # one point crossover
    def crossover(self, parent1, parent2):
        parent1_encoded = parent1.encode()
        parent2_encoded = parent2.encode()

        step = max(parent1.encode_step, parent2.encode_step)
        if len(parent1_encoded) != len(parent2_encoded):
            parent1_encoded = parent1_encoded.extend(step)
            parent2_encoded = parent2_encoded.extend(step)


        crossover_point = random.randint(2, (len(parent1_encoded) // step) - 1) * step
        
        child1_encoded = parent1_encoded[:crossover_point] + parent2_encoded[crossover_point:]
        child2_encoded = parent2_encoded[:crossover_point] + parent1_encoded[crossover_point:]
        
        child1 = CNNIndividual.decode(child1_encoded, step)
        child2 = CNNIndividual.decode(child2_encoded, step)
        
        return child1, child2

    def evaluate_individual(self, individual, x_val, y_val):
        individual.evaluate_fitness(x_val, y_val)
        fitness_categories = {
            'fitness_by_beta_1': 'beta_1',
            'fitness_by_beta_2': 'beta_2',
            'fitness_by_dropout_prob': 'dropout_prob',
            'fitness_by_learning_rate': 'learning_rate',
            'fitness_by_pooling_size': 'pooling_size',
            'fitness_by_stride': 'stride',
            'fitness_by_filter_size': 'filter_size',
            'fitness_by_filter_num': 'num_filters',
            'fitness_by_layer': 'num_conv_layers',
            'fitness_by_activation': 'activation',
            'fitness_by_padding': 'padding'
        }

        for category, attribute in fitness_categories.items():
            data = self.fitness_data[category]
            data.append({
                attribute: getattr(individual, attribute),
                'accuracy': individual.fitness
            })


    def evaluate_population_fitness(self, x_val, y_val):
        threads = []
        for individual in self.population:
            thread = Thread(target=self.evaluate_individual, args=(individual, x_val, y_val))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

    def fit_population(self, population, x_train, y_train, x_val, y_val, epochs, batch_size):
        threads = []
        for individual in population:
            thread = Thread(target=individual.construct_model, args=(x_train, y_train, x_val, y_val, epochs, batch_size))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()

    def populate(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=32, max_generations=30, tournament_size=3, selection_p=0.75,elite_percentage=10):
        if elite_percentage > 100:
            raise ValueError("elite_percentage must be between 0 and 100")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if epochs < 1:
            raise ValueError("epochs must be at least 1")
        if max_generations < 1:
            raise ValueError("max_generations must be at least 1")

        self.fit_population(self.population, x_train, y_train, x_val, y_val, epochs, batch_size)
        while self.generation + 1 < max_generations:
            self.evaluate_population_fitness(x_val, y_val)

            print(f"Generation {self.generation + 1} of {max_generations}. Best fitness: {max(self.population, key=lambda ind: ind.fitness).fitness}")
            self.best_individual_per_generation.append(max(self.population, key=lambda ind: ind.fitness))

            parents = self.selection(tournament_size, p=selection_p)

            num_elite_parents = int(len(parents) * elite_percentage / 100)
            elite_individuals = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)[:num_elite_parents]

            num_offspring_needed = self.population_size - len(elite_individuals)
            num_offspring = num_offspring_needed + num_offspring_needed % 2

            offspring = []
            for _ in range(num_offspring // 2):
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = self.crossover(parent1, parent2)
                offspring.extend([child1, child2])

            offspring = offspring[:num_offspring_needed]

            for individual in offspring:
                individual.mutate()
            
            self.fit_population(offspring, x_train, y_train, x_val, y_val, epochs, batch_size)

            self.population = offspring + elite_individuals
            self.generation += 1