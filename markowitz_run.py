def main():
    import numpy as np
    import pandas as pd
    from scipy.stats import gmean

    from markowitz import Markowitz

    # Leitura e pré-processamento dos dados de entrada:
    print('-'*50)
    print('Inicio da Leitura dos dados de entrada...')

    df = pd.read_excel('seriehistorica.xlsx', index_col = 0)
    df.fillna(method='ffill',inplace=True)

    pf = pd.read_excel('perfilfundos.xlsx', usecols = ['CNPJ', 'APLICACAO_MINIMA', 'CATEGORIA', 'SUBCATEGORIA'])
    pf['CNPJ'] = pf['CNPJ'].str.replace('[./-]', '', regex=True)
    pf['APLICACAO_MINIMA'] = pf['APLICACAO_MINIMA'].str.replace('-', '0').str.replace('R\$ ','').str.replace('.', '').astype(float)

    pf.set_index('CNPJ',inplace=True)
    pf.drop(index=(list(set(pf.index)-set(df.columns))),inplace=True)
    pf.reset_index(inplace=True)

    CNPJ_dict_tipos = dict()
    for cat in pf.CATEGORIA.unique():
        CNPJ_dict_tipos[cat] = pf.query('CATEGORIA == @cat').index.to_list()

    CNPJ_list = list(pf.CNPJ)
    df = df[CNPJ_list]

    pf.set_index('CNPJ',inplace=True)

    print('Dados carregados!')
    print()

    print('Preparando os parametros para o modelo...')
    # Preparando os parâmetros de entrada do modelo:
    df_retorno = df.pct_change().dropna()
    sigma = (df_retorno.cov()*252).to_numpy()
    media = (df_retorno+1).apply(gmean)**252-1

    valorMinFundos = np.array(pf['APLICACAO_MINIMA'])

    minRetorno = 0.002
    C = 100000
    K_min = 3
    K_max = 10
    P_min = 0.05
    P_max = 0.3

    P_categorias = {
        'A\u00E7\u00F5es': 0.25,
        'Cambial': 0.00,
        'Multimercados': 0.35,
        'Renda Fixa': 0.50
    }

    print('Parametros prontos!')
    print()

    # Instanciando o modelo e chamando o comando de "solve()":
    print('Instanciando o modelo para a minimizacao do risco...')

    modelo = Markowitz(C, CNPJ_list, CNPJ_dict_tipos, P_min, P_max, P_categorias, 
        K_min, K_max, valorMinFundos, sigma, media, minRetorno)

    print('Inicio da solucao do problema...')
    carteira = modelo.solve(time = 30)

    print('Processo de solucao completo!')
    print()

    print('Resultado obtido:')

    # Impressao dos resultados:
    carteira.exibir(C)
    print()

    print('Instanciando o modelo com o objetivo classico de Markowitz...')

    modelo = Markowitz(C, CNPJ_list, CNPJ_dict_tipos, P_min, P_max, P_categorias, K_min, K_max, 
        valorMinFundos, sigma, media, obj_type='markowitz', l=50)

    print('Inicio da solucao do problema...')
    carteira = modelo.solve(time = 30)    

    print('Processo de solucao completo!')
    print()

    print('Resultado obtido:')
    carteira.exibir(C)

    print('-'*50)

if __name__ == '__main__':
    main()
