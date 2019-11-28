import random
import numpy as np 

"""     
    Usando alguma biblioteca ou implementando diretamente em alguma linguagem de programação, desenvolva uma solução para minimizar a função de Rosenbrock (em anexo). 
    Faça testes pra identificar um conjunto de parâmetros adequado. Plote o gráfico de evolução da solução. 
    Adote :
    f(x,y) = 100*(y-x²)² + (1-x)²
    Xmin = Ymin = -2
    Xmax = Ymax = +2
    Mínimo global em x=y=+1

"""
random.seed(94)
class Particula():
    def __init__(self):
        self.posicao = np.array([(-1) ** (bool(random.getrandbits(1))) * random.random()*50, (-1)**(bool(random.getrandbits(1))) * random.random()*50])
        self.pbest_posicao = self.posicao
        self.pbest_valor = float('inf')
        self.velocidade = np.array([0,0])

    # def __str__(self):
        # print("Estou na posição ", self.posicao, " meu pbest é ", self.pbest_posicao)
    
    def move(self):
        self.posicao = self.posicao + self.velocidade

class Area():
    def __init__(self, target, min_max, n_particulas):
        self.target = target
        self.min_max = min_max
        self.n_particulas = n_particulas
        self.particulas = []
        self.gbest_valor = float('inf')
        self.gbest_posicao = np.array([random.random()*50, random.random()*50])

    def print_particulas(self):
        for particula in self.particulas:
            particula.__str__()
   
    def fitness(self, particula):
        """
            f(x,y) = 100*(y-x²)² + (1-x)²
        """
        x = particula.posicao[0]
        y = particula.posicao[1]
        fx = 100*(y-x**2)**2 + (1-x)**2
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

    def move_particulas(self, W, c1, c2):
        # W = 0.5
        # c1 = 0.8
        # c2 = 0.9 
        for particula in self.particulas:
            # global W
            new_velocidade = (W*particula.velocidade) + (c1*random.random()) * (particula.pbest_posicao - particula.posicao) + (random.random()*c2) * (self.gbest_posicao - particula.posicao)
            particula.velocidade = new_velocidade
            particula.move()

def plot_grafico(lista_pso):   
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    lista_posicoes = []
    lista_parametros = []
    lista_interacoes = []
    x = []
    y = []
    for item in lista_pso:
       lista_posicoes.append(item[0]) 
       x.append(item[0][0])
       y.append(item[0][1])
    #    lista_interacoes.append(item[1]) 
    #    lista_parametros.append(item[2])
    # plotting the points  
    # plt.plot(x, y)
    # plt.plot(x, y, label = str(item[2]) + "com " + str(item[1]))    
    # plt.title('Two lines on same graph!') 
    # plotting the points  
    # plt.plot(x, y, color='green', linestyle='dashed', linewidth = 3, 
    #         marker='o', markerfacecolor='blue', markersize=12)
    fig, ax  = plt.subplots()
    scatter  = ax.scatter(x, y, c=3)
    legenda  = ax.legend(*scatter.legend_elements(), loc="upper left", title="Classes")
    ax.add_artist(legenda)
    # show a legend on the plot 
    # plt.legend() 
    # plt.xlabel('x') 
    # plt.ylabel('y') 
    plt.show()
            
def main():
    """ W, c1 e c2 """
    lista_parametros1 = [0.5, 0.8, 0.9] 
    lista_parametros2 = [0.7, 0.5, 0.7]
    lista_parametros3 = [1, 0.4, 0.5]
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

    lista_resultados = []
    lista_resultados.append(PSO(lista_parametros1))
    lista_resultados.append(PSO(lista_parametros2))
    lista_resultados.append(PSO(lista_parametros3))
    plot_grafico(lista_resultados)

if __name__ == '__main__':
    main()
