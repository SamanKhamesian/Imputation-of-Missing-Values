import random

import numpy as np

from config import GAParam


class GeneticAlgorithm:
    def __init__(self, fcm, svr):
        self.cf = GAParam.CF
        self.mf = GAParam.MF
        self.generations = GAParam.GENERATIONS
        self.pop_size = GAParam.POPULATION_SIZE
        self.fcm = fcm
        self.svr = svr
        self.population = self.__generate_population()

    # Create random population
    def __generate_population(self):
        return [[random.randrange(2, 10), random.uniform(1.1, 10.0)] for _ in range(self.pop_size)]

    # Calculate fitness
    def __fitness(self, param):
        self.fcm.c = param[0]
        self.fcm.m = param[1]
        x = self.fcm.estimate_missing_values()
        y = self.svr.estimate_missing_value()
        f = np.power((x - y), 2).sum()
        return f

    # Select two parents randomly
    def __select_parents(self):
        index = random.sample(range(self.pop_size - 1), 2)
        parent_1 = self.population[index[0]]
        parent_2 = self.population[index[1]]

        del self.population[index[0]]
        del self.population[index[1]]

        return [parent_1, parent_2]

    # Recombine two parents and create two children
    def __crossover(self, chromosome_1, chromosome_2):
        prob = random.randrange(0, 101)
        if prob <= GAParam.CF * 100:
            index = random.randrange(0, 2)
            chromosome_1[index], chromosome_2[index] = chromosome_2[index], chromosome_1[index]

        return [chromosome_1, chromosome_2]

    # Mutate the given chromosome
    def __mutation(self, chromosome):
        prob = random.randrange(0, 101)
        if prob <= GAParam.MF * 100:
            index = random.randrange(0, 2)
            if index == 0:
                val = random.randrange(2, 10)
                chromosome[index] = val
            else:
                val = random.uniform(1.1, 10.0)
                chromosome[index] = val

        return chromosome

    def run(self):
        for _ in range(GAParam.GENERATIONS):
            parents = self.__select_parents()
            parents = self.__crossover(parents[0], parents[1])
            index = random.randrange(0, 2)
            parents[index] = self.__mutation(parents[index])
            self.population.append(parents[0])
            self.population.append(parents[1])

        self.population.sort(key=self.__fitness)
        return self.population[0][0], self.population[0][1]
