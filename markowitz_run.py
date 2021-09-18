def main():
    import numpy as np
    import pandas as pd
    
    from markowitz import Markowitz

    # Função auxiliar para o cálculo da média geométrica:
    def geo_mean(iterable):
        a = np.array(iterable)
        return a.prod()**(1.0/len(a))

    # Leitura e pré-processamento dos dados de entrada:
    print('-'*50)
    print('Inicio da Leitura dos dados de entrada...')

    df = pd.read_excel('seriehistorica.xlsx', index_col = 0)
    df.fillna(method='ffill',inplace=True)

    pf = pd.read_excel('perfilfundos.xlsx', usecols = ['CNPJ', 'APLICACAO_MINIMA', 'CATEGORIA', 'SUBCATEGORIA'])
    pf['CNPJ'] = pf['CNPJ'].str.replace('[./-]', '', regex=True)
    pf['APLICACAO_MINIMA'] = pf['APLICACAO_MINIMA'].str.replace('-', '0').str.replace('R\$ ','').str.replace('.', '').astype(float)
    pf.fillna(0, inplace=True)
    pf.set_index('CNPJ',inplace=True)

    pf_ordered = pf.sort_values(by ='CATEGORIA')
    pf_ordered.drop(index=(list(set(pf.index)-set(df.columns))),inplace=True)

    print('Dados carregados!')
    print()

    print('Preparando os parametros para o modelo...')
    # Preparando os parâmetros de entrada do modelo:
    categorias = list(pf_ordered['CATEGORIA'].unique())

    limites = [0]
    a = 0
    for c in categorias:
        a += len(pf_ordered[pf_ordered['CATEGORIA'] == c])
        limites.append(a)

    cnpj = list(pf_ordered.index)
    df = df[cnpj]

    df_retorno = df.pct_change().dropna()
    sigma = (df_retorno.cov()*252).to_numpy()
    media = (df_retorno+1).apply(geo_mean)**252-1

    minFundos = np.array(pf_ordered['APLICACAO_MINIMA'])
    n = len(minFundos)

    minRetorno = 0.002
    C = 100000

    P_categorias = [0.25, 0.15, 0.25, 0.35]

    K_min = 3
    K_max = 10
    P_min = 0.05
    P_max = 0.3

    print('Parametros prontos!')
    print()

    # Instanciando o modelo e chamando o comando de "solve()":
    print('Instanciando o modelo para a minimizacao do risco...')

    modelo = Markowitz(C, minRetorno, K_min, K_max, P_min, P_max, P_categorias, limites, n, minFundos, sigma, media, cnpj)

    print('Inicio da solucao do problema...')
    carteira = modelo.solve(time = 30)

    print('Processo de solucao completo!')
    print()

    print('Resultado obtido:')
    
    # Impressao dos resultados:
    carteira.exibir(C)
    print()

    print('Instanciando o modelo com o objetivo classico de Markowitz...')

    modelo = Markowitz(C, minRetorno, K_min, K_max, P_min, P_max, P_categorias, limites, n, minFundos, sigma, media, cnpj,
    obj_type='markowitz', l=40)

    print('Inicio da solucao do problema...')
    carteira = modelo.solve(time = 30)    

    print('Processo de solucao completo!')
    print()

    print('Resultado obtido:')
    carteira.exibir(C)

    print('-'*50)

if __name__ == '__main__':
    main()