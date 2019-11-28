import random
import numpy
import math

from deap import base
from deap import creator
from deap import tools
from deap import algorithms

"""

2) Resolver o problema do caixeiro viajante (representado pela imagem em anexo), sendo que a solução consiste 
   em encontrar um caminho sem repetições com custo mínimo. Execute o algoritmo com 3 variações de parâmetros 
   e plote o gráfico de evolução da aptidão. Interprete os resultados.
   Obs: Nesse problema, é preciso primeiro definir como representar um cromossomo e como definir uma função de aptidão.
 3012=11
 0123=11
 3210=11
 4123=11
 2143=11
 1234=11

"""

class AlgoritmoCaixeiroViajante():
    def __init__(self, tam_cidades):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.TAMANHO_CIDADES=tam_cidades
        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_permut", random.sample, range(tam_cidades), tam_cidades)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attr_permut)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.custo_viagem)
        self.toolbox.register("mate", tools.cxPartialyMatched)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def custo_entre_cidades(self, cidadeA, cidadeB):
        #custos possíveis
        custo = [[0,2,6,3], #custos saindo da cidade 1
                 [2,0,4,7], #custos saindo da cidade 2
                 [6,4,0,2], #custos saindo da cidade 3
                 [3,7,2,0]] #custos saindo da cidade 4
        return custo[cidadeA][cidadeB]
        
    def custo_viagem(self, individual): 
        #0=1
        #1=2
        #2=3 
        #3=4     
        custo = 0
        for i in range(self.TAMANHO_CIDADES):
            cidade_atual = individual[i]
            if i == self.TAMANHO_CIDADES - 1:
                prox_cidade = individual[0] #cidade origem
            else:
                prox_cidade = individual[i+1] #proxima cidade
            custo = custo + self.custo_entre_cidades(cidade_atual, prox_cidade)
        return custo,

    def plot_log(self, logbook):
        gen = logbook.select("gen")
        min = logbook.select("min")
        # avg = logbook.select("avg")
        max = logbook.select("max")

        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, min, "b-", label="Minimum Fitness")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness", color="b")
        for tl in ax1.get_yticklabels():
            tl.set_color("b")

        ax3 = ax1.twinx()
        line3 = ax3.plot(gen, max, "y-", label="Maximum Fitness")
        ax3.set_ylabel("Size")
        for tl in ax3.get_yticklabels():
            tl.set_color("y")
        lns = line1 + line3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="center right")

        plt.show()

    def print_ind(self, individual):
        #0=1
        #1=2
        #2=3 
        #3=4     
        lista = [1,1,1,1] #apenas para deixar com a numeração igual ao da imagem
        lista_melhores_roteiros = []
        for z, item in enumerate(individual.items): 
            custo = 0
            for indx in individual.items[z]:
                cidade_atual = individual.items[z][indx]
                if indx == len(individual.items[z]) - 1:
                    prox_cidade = individual.items[z][0] #cidade origem
                else:
                    prox_cidade = individual.items[z][indx+1] #proxima cidade
                custo = custo + self.custo_entre_cidades(cidade_atual, prox_cidade)
                melhor_roteiro = map(sum, zip(lista, individual.items[z]))
            lista_melhores_roteiros.append(list(melhor_roteiro))
        # melhor_roteiro = map(sum, zip(lista, individual.items[0]))
        print('Melhor roteiro: ', lista_melhores_roteiros)
        print('Menor Custo: '+str(custo))

def main():    
    """
     Evolutionary Algorithm Simple
     pop -> populacao
     toolbox -> definicoes previas
     cxpb -> probabilidade de um individuo sofrer crossover
     mutpb -> probabilidade de um individuo sofrer mutacao
     ngen -> numero de geracoes maxima
     stats -> forma de gerar as estatisticas, ja definido
     halloffame -> hall da fama escolhido
     verbose -> se True, imprime na tela a execucao do algoritmo
     pop, log = algorithms.eaSimple(pop, toolbox, cxpb=1, mutpb=0.1, ngen=10, stats=stats, halloffame=hof, verbose=True)
     
    """
    ClassCaixViaj = AlgoritmoCaixeiroViajante(4)
    random.seed(64)
    # pop = toolbox.population(n=40)
    hof = tools.HallOfFame(2)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # stats.register("avg", numpy.mean)
    # stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    cxpb_list = [0.1, 0.5, 1]
    mutpb_list = [0.01, 0.05, 0.1]
    ngen_list = [10, 20, 30]
    pop_list = [10, 20, 30]
    # cxpb_list = [0.1]
    # mutpb_list = [0.01]
    # ngen_list = [10]
    # pop_list = [10]

    for pop_item in pop_list:
        pop = ClassCaixViaj.toolbox.population(n=pop_item)
        for cxpb in cxpb_list:
            for mutpb in mutpb_list:
                for ngen in ngen_list:
                    pop, log = algorithms.eaSimple(pop, ClassCaixViaj.toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, stats=stats, halloffame=hof, verbose=False)
    #Plota o Gráfico
    ClassCaixViaj.plot_log(log)
    return ClassCaixViaj, hof

if __name__ == "__main__":
    ca, results = main()
    ca.print_ind(results)