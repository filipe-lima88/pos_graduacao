import random
import numpy as np 
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

"""     
    Otimizar uma rede neural artificial com PSO para o dataset IRIS.
"""
iris = datasets.load_iris()
x = iris.data[:, :2]
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.30)

random.seed(94)
class Particula():
    def __init__(self, paramMaxPos):
        # self.posicao = np.array([(-1) ** (bool(random.getrandbits(1))) * random.random()*50, (-1)**(bool(random.getrandbits(1))) * random.random()*50])
        self.posicao = np.array([(-1) ** (bool(random.getrandbits(1))) * random.random()*paramMaxPos, (-1)**(bool(random.getrandbits(1))) * random.random()*paramMaxPos])
        self.pbest_posicao = self.posicao
        self.pbest_valor = float('inf')
        self.velocidade = np.array([0,0])

    # def __str__(self):
        # print("Estou na posição ", self.posicao, " meu pbest é ", self.pbest_posicao)
    
    def move(self):
        self.posicao = self.posicao + self.velocidade

class Area():
    def __init__(self, target, min_max, n_particulas, paramMaxPos):
        self.target = target
        self.min_max = min_max
        self.n_particulas = n_particulas
        self.particulas = []
        self.gbest_valor = float('inf')
        # self.gbest_posicao = np.array([random.random()*50, random.random()*50])
        self.gbest_posicao = np.array([random.random()*paramMaxPos, random.random()*paramMaxPos])

    def print_particulas(self):
        for particula in self.particulas:
            particula.__str__()

    def fitness(self, particula):
        """
            f(x,y) = 100*(y-x²)² + (1-x)²
        """
        # x = particula.posicao[0]
        # y = particula.posicao[1]
        # fx = 100*(y-x**2)**2 + (1-x)**2
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

        return fx
        # return 100*(particle.position[0]-(particle.position[1]**2))**2 + (1-particle.position[1])**2

    def set_pbest(self):
        for particula in self.particulas:
            fitness_particula = self.fitness(particula)
            if(particula.pbest_valor > fitness_particula):
                particula.pbest_valor = fitness_particula
                particula.pbest_posicao = particula.posicao
            
    def set_gbest(self):
        for particula in self.particulas:
            best_fitness_particula = self.fitness(particula)
            if(self.gbest_valor > best_fitness_particula):
                self.gbest_valor = best_fitness_particula
                self.gbest_posicao = particula.posicao

    def move_particulas(self):
        W = 0.5
        c1 = 0.8
        c2 = 0.9 
        for particula in self.particulas:
            # global W
            new_velocidade = (W*particula.velocidade) + (c1*random.random()) * (particula.pbest_posicao - particula.posicao) + (random.random()*c2) * (self.gbest_posicao - particula.posicao)
            particula.velocidade = new_velocidade
            particula.move()

class RedeNeural():
    txAprendizadoDict = {0: 0.01, 1: 0.05, 2: 0.1, 3: 0.05}
    txMomentumDict    = {0: 0.01, 1: 0.05, 2: 0.1, 3: 1.0}
    cvScoresList = []
    cvScoresDict = {}
    varMaxIter   = 1000
    numeroIteracoes = None
    for numeroIteracoes in range(len(txAprendizadoDict)):
        txAprendizado = txAprendizadoDict[numeroIteracoes]
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
    #    clf = MLPClassifier(solver = 'sgd',
    #                        hidden_layer_sizes=(5,),
    #                        random_state=1,
    #                        learning_rate='constant',
    #                        learning_rate_init=txAprendizadoDict[numeroIteracoes],
    #                        max_iter=varMaxIter,
    #                        activation='logistic',
    #                        momentum=txMomentumDict[numeroIteracoes],
    #                        early_stopping=False,
    #                        validation_fraction=0.3,
    #                        tol=0.000001
    #                        )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    # cvScoresDict['taxa acerto'] = clf.score(x.iloc[test], y.iloc[test])

def main():
    def PSO(parametros):
        n_interacoes = 50
        lista_min_max = [-2, 2]
        n_particulas = 50
        target = 1 # Mínimo global em x=y=+1
        procura_area = Area(target, lista_min_max, n_particulas)
        vetor_particulas = [Particula() for _ in range(procura_area.n_particulas)]
        procura_area.particulas = vetor_particulas
        procura_area.print_particulas()

        interacao = 0
        while(interacao < n_interacoes):
            procura_area.set_pbest()    
            procura_area.set_gbest()

            if(abs(procura_area.gbest_valor - procura_area.target) <= (procura_area.min_max[0] and procura_area.min_max[1])):
                print("A melhor posição é ", procura_area.gbest_posicao, " com ", interacao, " interações \nParâmetros: ", parametros[0], ', ', parametros[1], ', ', parametros[2])
                procura_area.gbest_posicao
                break

            procura_area.move_particulas(parametros[0], parametros[1], parametros[2])
            interacao += 1
        return procura_area.gbest_posicao, interacao, parametros

    # plot_grafico(lista_resultados)

if __name__ == '__main__':
    main()
