import gurobipy as gp
import numpy as np
from math import ceil, sqrt

class Carteira(object):
    def __init__(self, Retorno, Risco, fundos, porcentagem, CNPJ_list, valorMinFundos):
        self.Retorno = Retorno
        self.Risco = Risco
        self.fundos = fundos
        self.porcentagem = porcentagem
        self.CNPJ_list = CNPJ_list
        self.valorMinFundos = valorMinFundos
       
    def exibir(self, C):
        print(f"Retorno esperado --> {round(100 * self.Retorno,2)}%")
        print(f"Risco estimado --> {round(100 * self.Risco,2)}%")
        for i in range(len(self.fundos)):
            #print(f"Fundo {self.fundos[i]} ({self.CNPJ_list[i]}) --> {round(100*self.porcentagem[i],2)}%; R$ {round(C*self.porcentagem[i],2)} --> R$ {self.valorMinFundos[i]}")
            print(f"Fundo {self.fundos[i]} ({self.CNPJ_list[i]}) \t Peso: {round(100*self.porcentagem[i],2)}% \t Valor: R$ {round(C*self.porcentagem[i],2)}")

    def __repr__(self):
        return ('''Carteira(Retorno={}, Risco={}, Qtd. de fundos={},
fundos={}, 
porcentagem={}, 
CNPJ_list={}, 
valorMinFundos={}'''.format(self.Retorno, self.Risco, len(self.fundos), self.fundos, self.porcentagem, self.CNPJ_list, self.valorMinFundos))

    def __str__(self):
        return self.__repr__()

class Markowitz(object):
    def __init__(self, C, CNPJ_list, CNPJ_dict_tipos, P_min, P_max, P_categorias, K_min, K_max, 
    valorMinFundos, sigma, media, minRetorno = 0, obj_type = 'Risco', l = 1):
        self.C = C
        self.minRetorno = minRetorno
        self.P_min = P_min
        self.P_max = P_max

        self.K_min = max(K_min, ceil(1 / P_max))
        self.K_max = min(K_max, ceil(1 / P_min))
        self.P_categorias = P_categorias
        self.CNPJ_list = CNPJ_list
        self.CNPJ_dict_tipos = CNPJ_dict_tipos
        self.n = len(self.CNPJ_list)

        for key in self.P_categorias.keys():
            if( (self.P_categorias[key] < self.P_min) and (self.P_categorias[key] > 0) ):
                self.P_categorias[key] = self.P_min

        self.valorMinFundos = valorMinFundos
        self.categorias = list(self.CNPJ_dict_tipos.keys())
        self.sigma = sigma
        self.media = np.array(media)
        self.l = l

        self.GAP = float('inf')
        self.sol_time = 0.0

        try:
            self.obj_type = str(obj_type).lower()
        except:
            self.obj_type = 'Risco'

        self.model = gp.Model()

        self.w = self.model.addMVar(self.n, ub = 1, vtype = gp.GRB.CONTINUOUS)
        self.y = self.model.addMVar(self.n, vtype = gp.GRB.BINARY)

        self.modelo_classico_labels = set(['classico', 'markowitz'])

        if(self.obj_type in self.modelo_classico_labels):
            self.obj_fun = self.model.setObjective(self.w @ self.media - self.l * (self.w @ self.sigma @ self.w), sense = gp.GRB.MAXIMIZE)
        else:
            self.obj_fun = self.model.setObjective(self.w @ self.sigma @ self.w, sense = gp.GRB.MINIMIZE)
            self.c1 = self.model.addConstr((self.w @ self.media) >= self.minRetorno)
    
        self.c2 = self.model.addConstr(self.w.sum() == 1)
        self.c3 = self.model.addConstrs(self.w[i] >= max(self.P_min, self.valorMinFundos[i]/self.C) * self.y[i] for i in range(self.n))
        self.c4 = self.model.addConstrs(self.w[i] <= self.P_max * self.y[i] for i in range(self.n))
        self.c5 = self.model.addConstr(self.y.sum() >= self.K_min)
        self.c6 = self.model.addConstr(self.y.sum() <= self.K_max)
        if(  round(sum(list(self.P_categorias.values())), 5) == 1  ):
            self.c7 = self.model.addConstrs(self.w[self.CNPJ_dict_tipos[k]].sum() == self.P_categorias[k] for k in self.categorias)
        else:
            self.c7 = self.model.addConstrs(self.w[self.CNPJ_dict_tipos[k]].sum() <= self.P_categorias[k] for k in self.categorias)
        
        self.result = None
    
    def solve(self, time=None, heur=None, log=0):
        if(self.K_min > self.K_max):
            print('Inconsistencia nos valores de K_min VS K_max!')
            return Carteira(0, 0, [], [], [], [])

        if(self.P_min > self.P_max):
            print('Inconsistencia nos valores de P_min VS P_max!')
            return Carteira(0, 0, [], [], [], [])

        if(  round(sum(list(self.P_categorias.values())), 5) < 1  ):
            print('Problema nos percentuais minimos para cada tipo de fundo!')
            return Carteira(0, 0, [], [], [], [])

        if(time != None):
            self.model.setParam('TimeLimit', time)
        if(heur != None):
            self.model.setParam('Heuristics', heur)
        if(log >= 0):
            try:
                self.model.Params.LogToConsole = log
            except:
                self.model.Params.LogToConsole = 1
        else:
            self.model.Params.LogToConsole = 1

        self.result = self.model.optimize()

        try:
            self.w[0].X >= 0
        except:
            print('Nenhuma solucao encontrada!')
            return Carteira(0, 0, [], [], [], [])
        
        self.Retorno = self.w.X @ self.media
        if (self.obj_type in self.modelo_classico_labels):
            self.Risco = sqrt(self.w.X @ self.sigma @ self.w.X)
        else:
            self.Risco = sqrt(self.model.getObjective().getValue())

        self.fundos = np.flatnonzero(self.w.X != 0).tolist()
        self.porcentagem = [self.w[j].X[0] for j in self.fundos]
        self.cnpj_escolhidos = [self.CNPJ_list[i] for i in self.fundos]
        self.apm_escolhidos = [self.valorMinFundos[i] for i in self.fundos]

        self.GAP = self.model.MIP_GAP
        self.sol_time = self.model.Runtime

        return Carteira(self.Retorno, self.Risco, self.fundos, self.porcentagem, self.cnpj_escolhidos, self.apm_escolhidos)

    def solve_fronteira(self, minRetornoInit, minRetornoFinal, tam_passo, taxa_livre_de_risco = 0.0525, 
    time = 15, heur = None, log = 0, print_steps = False):
        self.update_minRetorno(minRetornoInit)
        
        self.lista_retornos = list()
        self.lista_riscos = list()
        self.lista_sharpes = list()
        self.lista_carteiras = list()
        self.tam_passo = tam_passo

        if(self.tam_passo <= 0):
            self.tam_passo = 0.05
            print('Tamanho do passo automaticamente ajustado para:', self.tam_passo)

        self.carteira = Carteira(0, 0, [], [], [], [])

        while self.minRetorno <= minRetornoFinal:
            if(print_steps):
                print('minRetorno atual:', self.minRetorno)
            self.carteira = self.solve(time, heur, log)
            if(self.carteira.Retorno != 0):
                self.lista_retornos.append(self.carteira.Retorno)
                self.lista_riscos.append(self.carteira.Risco)
                self.lista_sharpes.append((self.carteira.Retorno - taxa_livre_de_risco)/self.carteira.Risco)
            else:
                break
            self.lista_carteiras.append(self.carteira)
            self.minRetorno += self.tam_passo
            self.update_minRetorno(self.minRetorno)

        self.max_sharpe = np.argmax(self.lista_sharpes)

        return self.lista_carteiras, self.max_sharpe    

    @property
    def P_min(self):
        return self._P_min

    @property
    def P_max(self):
        return self._P_max

    @P_min.setter
    def P_min(self, value):
        self._P_min = min(abs(value), 1)

    @P_max.setter
    def P_max(self, value):
        self._P_max = min(abs(value), 1)

    def update_minRetorno(self, minRetorno = 0):
        self.minRetorno = minRetorno
        try:
            self.model.remove(self.c1)
        except:
            pass
        self.c1 = self.model.addConstr((self.w @ self.media) >= self.minRetorno)
        
    def exibir_par(self):

        print('Dados considerados para a otimizacao:')
        print(f'Capital = {self.C}')
        print(f'Retorno minimo especificado = {self.minRetorno}')
        print(f'P_min = {100*self.P_min}%')
        print(f'P_max = {100*self.P_max}%')
        print(f'K_min = {self.K_min}')
        print(f'K_max = {self.K_max}')
        print(f'P_categorias = {self.P_categorias}')
    
