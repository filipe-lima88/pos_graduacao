import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from random import randint

"""
1) Encontrar valor de x para o qual a função f(x) = x² - 3x + 4 assume o valor mínimo
- Assumir que x ∈ [-10, +10]
- Codificar X como vetor binário
- Criar uma população inicial com 4 indivíduos
- Aplicar Mutação com taxa de 1%
- Aplicar Crossover com taxa de 60%
- Usar seleção por roleta.
- Usar no máximo 5 gerações.
"""
class AlgoritmoGenetico():
    def __init__(self, x_min, x_max, tam_populacao, taxa_mutacao, taxa_crossover, num_geracoes):
        # inicializa todos os atributos q servirao de parametros para o problema
        self.x_min = x_min
        self.x_max = x_max
        self.tam_populacao = tam_populacao
        self.taxa_mutacao = taxa_mutacao
        self.taxa_crossover = taxa_crossover
        self.num_geracoes = num_geracoes
        # calcula o número de bits do x_min e x_max no formato binário com sinal
        qtd_bits_x_min = len(bin(x_min).replace('0b', '' if x_min < 0 else '+'))
        qtd_bits_x_max = len(bin(x_max).replace('0b', '' if x_max < 0 else '+'))
        # o maior número de bits representa o número de bits a ser utilizado para gerar individuos
        self.num_bits = qtd_bits_x_max if qtd_bits_x_max >= qtd_bits_x_min else qtd_bits_x_min
        self.gerar_populacao()

    def gerar_populacao(self):
        self.populacao = [[] for i in range(self.tam_populacao)]
        random.seed(64)
        for individuo in self.populacao:
            # para cada individuo sorteia números entre "x_min" e "x_max"
            num = random.randint(self.x_min, self.x_max)            
            # converte o número sorteado para binário com sinal
            num_bin = bin(num).replace('0b', '' if num < 0 else '+').zfill(self.num_bits)
            print(num, num_bin)
            # transforma o número binário em vetor
            for bit in num_bin:
                individuo.append(bit)

    def funcao_objetivo(self, num_bin):
        # converte o número binário para inteiro
        num = int(''.join(num_bin), 2)
        # retorna o resultado da função objetivo
        return num**2 -3*num + 4

    def avaliar(self):
        self.avaliacao = []
        for individuo in self.populacao:
            self.avaliacao.append(self.funcao_objetivo(individuo))

    def selecionar(self):
        from operator import attrgetter
        import random
        # Cria lista de prob de seleção para minimização sum(Fobj)/Fobj
        lista_prob_selecao_minimizacao = list(sum(self.avaliacao)/x for x in self.avaliacao)
        y = 0
        lista_prob_roleta = []
        # Cria lista de valores por range para a roleta
        for x in lista_prob_selecao_minimizacao:
            y += x
            lista_prob_roleta.append(y)
        pop_fitness = zip(self.populacao, lista_prob_roleta)  
        # Ordena do menor valor para o maior
        s_inds = sorted(pop_fitness, key=lambda x: x[1], reverse=False) 
        # Gera número aleatório entre 0 e valor maximo
        valor_moeda = random.uniform(0, max(lista_prob_roleta))
        # Seleciona o individuo com base no valor roletado entre o range de valores 
        for i in range(1):
            valor_ind_anterior = 0
            for ind in zip(s_inds): 
                if (valor_ind_anterior <= valor_moeda) and (valor_moeda <= ind[0][1]):
                    escolhido = ind[0][0]
                    break
                valor_ind_anterior = ind[0][1]

        return escolhido

    def crossover(self, pai, mae):
        if randint(1,100) <= self.taxa_crossover:
            ponto_de_corte = randint(1, self.num_bits - 1)
            filho_1 = pai[:ponto_de_corte] + mae[ponto_de_corte:]
            filho_2 = mae[:ponto_de_corte] + pai[ponto_de_corte:]  
        else:
            filho_1 = pai[:]
            filho_2 = mae[:]

        return (filho_1, filho_2)

    def mutar(self, individuo):
        # cria a tabela com as regras de mutação
        tabela_mutacao = str.maketrans('+-01', '-+10')
        # caso a taxa de mutação seja atingida, ela é realizada em um bit aleatório
        if randint(1,100) <= self.taxa_mutacao:
            bit = randint(0, self.num_bits - 1)
            individuo[bit] = individuo[bit].translate(tabela_mutacao)

    def encontrar_filho_mais_apto(self):
        candidatos = zip(self.populacao, self.avaliacao)
        # seleciona o individuo com melhor avaliacao, nesse caso de minimizacao o menor valor
        individuo_selecionado = min(candidatos, key=lambda elemento: elemento[1])
        lista_msg = []
        lista_msg.append('Indivíduo: {}'.format(individuo_selecionado[0]))
        lista_msg.append('Valor de X: '+ str(int(''.join(individuo_selecionado[0]), 2)))
        lista_msg.append('Valor de f(X): '+ str(individuo_selecionado[1]))
    
        return lista_msg

def main():
    # (x_min, x_max, tam_populacao, taxa_mutacao, taxa_crossover, num_geracoes):
    algoritmo_genetico = AlgoritmoGenetico(-10, 10, 4, 1, 60, 5)
    # avalia a população inicial
    algoritmo_genetico.avaliar()
    for i in range(algoritmo_genetico.num_geracoes):
        # imprime o resultado a cada geração
        print( 'Resultado: Geração {} -> {}'.format(i, algoritmo_genetico.encontrar_filho_mais_apto()) )
        # cria uma nova população
        nova_populacao = []
        while len(nova_populacao) < algoritmo_genetico.tam_populacao:
            pai = algoritmo_genetico.selecionar()
            mae = algoritmo_genetico.selecionar()
            filho_1, filho_2 = algoritmo_genetico.crossover(pai, mae)
            algoritmo_genetico.mutar(filho_1)
            algoritmo_genetico.mutar(filho_2)
            nova_populacao.append(filho_1)
            nova_populacao.append(filho_2)

        algoritmo_genetico.populacao = nova_populacao
        algoritmo_genetico.avaliar()

    # procura o filho mais apto dentro da população e exibe o resultado do algoritmo genético
    print('Resultado: Geração {} -> {}'.format(i+1, algoritmo_genetico.encontrar_filho_mais_apto()) )

if __name__ == '__main__':
    main()