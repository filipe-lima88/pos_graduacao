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

# from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.metrics import accuracy_score

"""     
    Otimizar os parâmetros de uma rede neural com PSO.
"""

# dict_activation = {1:"identity", 2:"logistic", 3:"tahn", 4:"relu" }
# dict_solver = {1:"lbfgs", 2:"sgd", 3:"adam"}

iris = datasets.load_digits()
x = iris.data
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.30)

particle_min = 0.001
particle_max = 1 
dict_learning_rate = {1:"constant", 2:"invscaling", 3:"adaptive"}

random.seed(100)

class PSO():
    def generate(self, size, learning_rate_init_min, learning_rate_init_max, 
                 momentum_min, momentum_max, learning_rate_min, learning_rate_max, 
                 s_learning_rate_init_min, s_learning_rate_init_max, s_momentum_min, s_momentum_max):
        learning_rate_init = random.uniform(learning_rate_init_min, learning_rate_init_max)
        momentum = random.uniform(momentum_min, momentum_max)
        learning_rate = random.randrange(learning_rate_min, learning_rate_max)

        part = creator.Particle([learning_rate_init, momentum, learning_rate])

        part.learning_rate_init_min = learning_rate_init_min
        part.learning_rate_init_max = learning_rate_init_max
        part.momentum_min = momentum_min
        part.momentum_max = momentum_max
        part.learning_rate_min = learning_rate_min
        part.learning_rate_max = learning_rate_max  

        part.speed = [random.uniform(s_learning_rate_init_min, s_learning_rate_init_max),random.uniform(s_momentum_min, s_momentum_max)]
        return part

    def updateParticle(self, part, best, c1, c2):
        u1 = (random.uniform(0, c1) for _ in range(len(part)))
        u2 = (random.uniform(0, c2) for _ in range(len(part)))
        v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
        v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
        part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
        # gera um peso de inercia, entre 0.4 e 0.9
        inertia_weights = (random.uniform(0.4, 0.9) for _ in range(len(part)))
        # aplica os pesos de inercia a velocidade
        part.speed = list(map(operator.mul, inertia_weights, part.speed))
        # atualiza a posição
        part[:2] = list(map(operator.add, part, part.speed))

        # impede a particula de ser ajustada para fora do espaço de busca
        if part[0] < part.learning_rate_init_min:
            part[0] = part.learning_rate_init_min
        elif part[0] > part.learning_rate_init_max:
            part[0] = part.learning_rate_init_max
        
        if part[1] < part.momentum_min:
            part[1] = part.momentum_min
        elif part[1] > part.momentum_max:
            part[1] = part.momentum_max

    def __init__(self):
        creator.create("Fitness", base.Fitness, weights=(0.1,))
        creator.create("Particle", list, 
                       fitness=creator.Fitness, 
                       learning_rate_init_min=None, 
                       learning_rate_init_max=None, 
                       momentum_min=None, 
                       momentum_max=None,
                       learning_rate_min=None,
                       learning_rate_max=None,
                       speed=list, 
                       s_learning_rate_init_min=None, 
                       s_learning_rate_init_max=None,
                       s_momentum_min=None, 
                       s_momentum_max=None,  
                       s_learning_rate_min=None,
                       s_learning_rate_max=None,
                       best=None)
        self.toolboxPSO = base.Toolbox()
        self.toolboxPSO.register("particle", self.generate, size=3, 
                            learning_rate_init_min=particle_min,
                            learning_rate_init_max=particle_max,
                            momentum_min=particle_min,
                            momentum_max=particle_max,
                            learning_rate_min=1,
                            learning_rate_max=3,
                            s_learning_rate_init_min=particle_min*0.01, 
                            s_learning_rate_init_max=particle_max*0.01, 
                            s_momentum_min=particle_min*0.01, 
                            s_momentum_max=particle_max*0.01)
        self.toolboxPSO.register("population", tools.initRepeat, list, self.toolboxPSO.particle)
        self.toolboxPSO.register("update", self.updateParticle, c1=2.0, c2=2.0)
        self.toolboxPSO.register("evaluate", self.evaluation)

    def evaluation(self, individuo):
        param0 = individuo[0]
        param1 = individuo[1]
        param2 = individuo[2]
        clf = MLPClassifier(solver='sgd', 
                            alpha=1e-5, 
                            hidden_layer_sizes=(5,), 
                            random_state=1,
                            learning_rate=dict_learning_rate[param2], 
                            learning_rate_init=param0, 
                            momentum=param1,
                            max_iter=100,
                            early_stopping=False,
                            activation='logistic',
                            validation_fraction=0.3,
                            tol=0.000001,
                            verbose=False)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        score = accuracy_score(y_test, y_pred)
        return score, 

def main():
    pso = PSO() 
    pop = pso.toolboxPSO.population(n=20)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "evals"] + stats.fields

    GEN = 20
    best = None

    for g in range(GEN):
        for part in pop:
            part.fitness.values = pso.toolboxPSO.evaluate(part)
            if not part.best or part.best.fitness < part.fitness:
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not best or best.fitness < part.fitness:
                best = creator.Particle(part)
                best.fitness.values = part.fitness.values
        for part in pop:
            pso.toolboxPSO.update(part, best)

        # Gather all the fitnesses in one list and print the stats
        logbook.record(gen=g, evals=len(pop), **stats.compile(pop))
        print(logbook.stream)
    
    return pso, pop, logbook, best

if __name__ == "__main__":
    resultsPSO = main()
    print('Melhores parâmetros para a MLP => learning_rate_init, momentum e learning_rate : ',resultsPSO[3])
    print('Fitness(Acurácia) do melhor resultado PSO: ',resultsPSO[0].evaluation(resultsPSO[3]))