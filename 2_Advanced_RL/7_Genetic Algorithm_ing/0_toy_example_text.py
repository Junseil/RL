import random
from matplotlib import pyplot as plt

spawn_generation = ''
for i in range(ord('A'), ord('z')+1):
    spawn_generation += chr(i)
target = 'Hello_Python3'

class Individual:
    def __init__(self, string, fitness=0):
        self.string = string
        self.fitness = fitness

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def spawn_population(length=26, size=100):
    tmp = []
    for _ in range(size):
        string = ''.join(random.choices(spawn_generation, k=length))
        individual = Individual(string)
        tmp.append(individual)
    return tmp

def recombine(p1_, p2_):
    p1 = p1_.string
    p2 = p2_.string
    child1 = []
    child2 = []
    cross_pt = random.randint(0, len(p1))
    child1.extend(p1[0:cross_pt])
    child1.extend(p2[cross_pt:])
    child2.extend(p2[0:cross_pt])
    child2.extend(p1[cross_pt:])
    c1 = Individual(''.join(child1))
    c2 = Individual(''.join(child2))
    return c1, c2

def mutate(x, mut_rate=0.01):
    new_x_ = []
    for char in x.string:
        if random.random() < mut_rate:
            new_x_.extend(random.choices(spawn_generation, k=1))
        else:
            new_x_.append(char)
    new_x = Individual(''.join(new_x_))
    return new_x

def evaluate_population(tmp, target):
    avg_fit = 0
    for i in range(len(tmp)):
        fit = similar(tmp[i].string, target)
        tmp[i].fitness = fit
        avg_fit = fit
    avg_fit /= len(tmp)
    return tmp, avg_fit

def next_generation(tmp, size=100, length=26, mut_rate=0.01):
    new_tmp = []
    while len(new_tmp) < size:
        parents = random.choices(tmp, k=2, weights=[x.fitness for x in tmp])
        offspring_ = recombine(parents[0], parents[1])
        child1 = mutate(offspring_[0], mut_rate=mut_rate)
        child2 = mutate(offspring_[1], mut_rate=mut_rate)
        offspring = [child1, child2]
        new_tmp.extend(offspring)
    return new_tmp

num_generations = 150
population_size = 900
str_len = len(target)
mutation_rate = 0.00001

pop_fit = []
pop = spawn_population(size=population_size, length=str_len)
for gen in range(num_generations):
    pop, avg_fit = evaluate_population(pop, target)
    pop_fit.append(avg_fit)
    new_pop = next_generation(pop, size=population_size, length=str_len,
                              mut_rate=mutation_rate)
    pop = new_pop
    
pop.sort(key=lambda x: x.fitness, reverse=True)
print(pop[0].string)