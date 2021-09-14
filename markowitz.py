import gurobipy as gp
import numpy as np
from math import ceil

class Carteira(object):
    def __init__(self, Retorno, Risco, fundos, porcentagem, descricao, minFundos):
        self.Retorno = Retorno
        self.Risco = Risco
        self.fundos = fundos
        self.porcentagem = porcentagem
        self.descricao = descricao
        self.minFundos = minFundos
       
    def exibir(self, C):
        print(f"Retorno esperado --> {round(100 * self.Retorno,2)}%")
        print(f"Risco estimado --> {round(100 * self.Risco,2)}%")
        for i in range(len(self.fundos)):
            print(f"Fundo {self.fundos[i]} ({self.descricao[i]}) --> {round(100*self.porcentagem[i],2)}%; R$ {round(C*self.porcentagem[i],2)} --> R$ {self.minFundos[i]}")

    def __repr__(self):
        return ('''Carteira(Retorno={}, Risco={}), Qtd. de fundos={}, porcentagem={}'''.format(
            self.Retorno, self.Risco, len(self.fundos), self.porcentagem))

    def __str__(self):
        return ('Um objeto Carteira com {} fundos'.format(len(self.fundos)))

class Markowitz(object):

    def __init__(self, C, minRetorno, K_min, K_max, P_min, P_max, P_categorias, limites, 
        n, minFundos, sigma, media, descricao, obj_type = 'Risco', l = 1, log = 0):
        self.C = C
        self.minRetorno = minRetorno
        self.P_min = P_min
        self.P_max = P_max

        self.K_min = max(K_min, ceil(1 / P_max))
        self.K_max = min(K_max, ceil(1 / P_min))
        self.P_categorias = P_categorias
        self.limites = limites
        self.n = n
        self.log = log

        for idx in range(len(self.P_categorias)):
            if( (self.P_categorias[idx] < self.P_min) and (self.P_categorias[idx] > 0) ):
                self.P_categorias[idx] = self.P_min

        self.n = n
        self.minFundos = minFundos
        self.sigma = sigma
        self.media = media
        self.descricao = descricao
        
        self.l = l
        try:
            self.obj_type = str(obj_type).lower()
        except:
            self.obj_type = 'Risco'

        self.model = gp.Model()
        self.model.Params.LogToConsole = self.log

        self.w = self.model.addVars(range(n), ub = 1, vtype = gp.GRB.CONTINUOUS)
        self.y = self.model.addVars(range(n), vtype = gp.GRB.BINARY)

        self.modelo_classico_labels = set(['classico', 'markowitz'])

        if(self.obj_type in self.modelo_classico_labels):
            self.obj_fun = self.model.setObjective(gp.quicksum(self.media[i] * self.w[i] for i in range(self.n)) - 
            self.l * gp.quicksum(self.w[i] * self.sigma[i][j] * self.w[j] for i in range(self.n) for j in range(self.n)), 
            sense = gp.GRB.MAXIMIZE)
        else:
            self.obj_fun = self.model.setObjective(gp.quicksum(
                self.w[i] * self.sigma[i][j] * self.w[j] for i in range(self.n) for j in range(self.n)), sense = gp.GRB.MINIMIZE)
            self.c1 = self.model.addConstr(gp.quicksum(self.media[i] * self.w[i] for i in range(self.n)) >= self.minRetorno)
    
        self.c2 = self.model.addConstr(gp.quicksum(self.w[i] for i in range(self.n)) == 1)
        self.c3 = self.model.addConstrs(self.w[i] >= max(self.P_min, self.minFundos[i]/self.C) * self.y[i] for i in range(self.n))
        self.c4 = self.model.addConstrs(self.w[i] <= self.P_max * self.y[i] for i in range(self.n))
        self.c5 = self.model.addConstr(gp.quicksum(self.y[i] for i in range(self.n)) >= self.K_min)
        self.c6 = self.model.addConstr(gp.quicksum(self.y[i] for i in range(self.n)) <= self.K_max)
        self.c7 = self.model.addConstrs(gp.quicksum(self.w[i] for i in range(self.limites[j] , self.limites[j+1])) <= 
            self.P_categorias[j] for j in range(len(self.limites)-1))
    
    def solve(self, time=None, heur=None):
        if(self.K_min > self.K_max):
            print('Inconsistencia nos valores de K_min VS K_max!')
            return Carteira(0, 0, [], 0, [], [])

        if(self.P_min > self.P_max):
            print('Inconsistencia nos valores de P_min VS P_max!')
            return Carteira(0, 0, [], 0, [], [])

        if(  round(sum(self.P_categorias), 5) < 1  ):
            print('Problema nos percentuais minimos para cada tipo de fundo!')
            return Carteira(0, 0, [], 0, [], [])

        if(time != None):
            self.model.setParam('TimeLimit', time)
        if(heur != None):
            self.model.setParam('Heuristics', heur)

        self.result = self.model.optimize()

        try:
            self.w[0].X >= 0
        except:
            print('Nenhuma solucao encontrada!')
            return Carteira(0, 0, [], 0, [], [])
        
        self.Retorno = np.dot(self.media, np.array([self.w[i].x for i in range(len(self.w))]))
        if (self.obj_type in self.modelo_classico_labels):
            self.vect_w = np.array([[self.w[i].x for i in range(self.n)]]).T
            self.Risco = float(np.sqrt(self.vect_w.T @ self.sigma @ self.vect_w))
        else:
            self.Risco = np.sqrt(self.model.getObjective().getValue())

        self.fundos = [i for i in range(len(self.w)) if (self.w[i].x) != 0.0 ]
        self.porcentagem = [self.w[i].x for i in self.fundos]
        self.cnpj_escolhidos = [self.descricao[i] for i in self.fundos]
        self.apm_escolhidos = [self.minFundos[i] for i in self.fundos]

        return Carteira(self.Retorno, self.Risco, self.fundos, self.porcentagem, self.cnpj_escolhidos, self.apm_escolhidos)

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
        self.c1 = self.model.addConstr(gp.quicksum(self.media[i] * self.w[i] for i in range(self.n)) >= self.minRetorno)
        
    def exibir_par(self):

        print('Dados considerados para a otimizacao:')
        print(f'Capital = {self.C}')
        print(f'Retorno minimo especificado = {self.minRetorno}')
        print(f'P_min = {100*self.P_min}%')
        print(f'P_max = {100*self.P_max}%')
        print(f'K_min = {self.K_min}')
        print(f'K_max = {self.K_max}')
        print(f'P_categorias = {self.P_categorias}')
    
