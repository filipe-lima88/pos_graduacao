import operator
import random

import numpy
import math

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

"""     
    Otimizar uma rede neural artificial com PSO para o dataset IRIS.
"""

dict_activation = {1:"identity", 2:"logistic", 3:"tahn", 4:"relu" }
dict_solver = {1:"lbfgs", 2:"sgd", 3:"adam"}
dict_learning_rate = {1:"constant", 2:"invscaling", 3:"adaptive"}

list_parametros = []
list_parametros.append(dict_activation)
list_parametros.append(dict_solver)
list_parametros.append(dict_learning_rate)


iris = datasets.load_iris()
x = iris.data[:, :2]
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.30)

class PSO():
    def generate(size, pmin, pmax, smin, smax):
        part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
        part.speed = [random.uniform(smin, smax) for _ in range(size)]
        part.smin = smin
        part.smax = smax
        return part

    def updateParticle(part, best, phi1, phi2):
        u1 = (random.uniform(0, phi1) for _ in range(len(part)))
        u2 = (random.uniform(0, phi2) for _ in range(len(part)))
        v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
        v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
        part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
        for i, speed in enumerate(part.speed):
            if abs(speed) < part.smin:
                part.speed[i] = math.copysign(part.smin, speed)
            elif abs(speed) > part.smax:
                part.speed[i] = math.copysign(part.smax, speed)
        part[:] = list(map(operator.add, part, part.speed))

    def fitness_mlp(self):
        clf = MLPClassifier(solver='sgd', 
                    alpha=1e-5, 
                    hidden_layer_sizes=(5,), 
                    random_state=1,
                    learning_rate='adaptive', #constant
                    learning_rate_init=txAprendizadoDict[numeroIteracoes], 
                    momentum=txMomentumDict[numeroIteracoes],
                    max_iter=varMaxIter,
                    early_stopping=False,
                    activation='logistic',
                    validation_fraction=0.3,
                    tol=0.000001)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        return y_pred

    def __init__(self, aTamanho, aPMinMax, aSMinMax):
        # self.posicao = np.array([(-1) ** (bool(random.getrandbits(1))) * random.random()*50, (-1)**(bool(random.getrandbits(1))) * random.random()*50])
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, smin=None, smax=None, best=None)
        toolbox = base.Toolbox()
        toolbox.register("particle", generate, size=aTamanho, pmin=-aPMinMax, pmax=aPMinMax, smin=-aSMinMax, smax=aSMinMax)
        toolbox.register("population", tools.initRepeat, list, toolbox.particle)
        toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
        toolbox.register("evaluate", fitness_mlp)

def main():
    pso = PSO(50, ) 
    pop = pso.toolbox.population(n=5)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    GEN = 10
    best = None

    for g in range(GEN):
        for part in pop:
            part.fitness.values = toolbox.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            toolbox.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)
    
    return pop, logbook, best

if __name__ == "__main__":
    main()