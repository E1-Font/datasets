
import numpy as np
import pandas as pd


def gerar_dataset(grr, n=1000):
    """
    Gera um dataset de regressão para predição da conversão de um reator químico.

    Parâmetros
    ----------
    grr : int ou str
        GRR do aluno. Controla o tipo de reação, presença de catalisador,
        nível de ruído e quantidade de valores faltantes.

    n : int
        Número de amostras.

    Retorna
    -------
    pandas.DataFrame
    """

    grr = str(grr)

    if len(grr) < 4 or not grr.isdigit():
        raise ValueError("O GRR deve possuir pelo menos quatro dígitos.")

    rng = np.random.default_rng(int(grr))

    d1 = int(grr[-1])   # tipo de reação
    d2 = int(grr[-2])   # catalisador
    d3 = int(grr[-3])   # ruído
    d4 = int(grr[-4])   # missing values

    # --------------------------------------------------------
    # Tipo de reação
    # --------------------------------------------------------

    reacao_ab = d1 < 5

    # --------------------------------------------------------
    # Processo catalítico
    # --------------------------------------------------------

    usa_catalisador = d2 < 5

    # --------------------------------------------------------
    # Variáveis de entrada
    # --------------------------------------------------------

    temperatura = rng.uniform(320, 650, n)          # K
    pressao = rng.uniform(1, 20, n)                 # bar
    tempo = rng.uniform(0.5, 15, n)                 # min
    conc_a = rng.uniform(0.2, 2.0, n)               # mol/L

    if reacao_ab:
        conc_b = None
    else:
        conc_b = rng.uniform(0.2, 2.0, n)

    if usa_catalisador:
        catalisador = rng.uniform(0.5, 5.0, n)
    else:
        catalisador = np.zeros(n)

    # --------------------------------------------------------
    # Modelo cinético simplificado
    # --------------------------------------------------------

    R = 8.314

    Ea = 52000 if reacao_ab else 60000

    k = 3e6 * np.exp(-Ea / (R * temperatura))

    # influência da pressão

    if usa_catalisador:
        fator_pressao = 1 + 0.02 * (pressao - 10)
        fator_cat = 1 + 0.35 * catalisador
    else:
        fator_pressao = 1 + 0.10 * (pressao - 10)
        fator_cat = 1.0

    if reacao_ab:
        velocidade = (
            k
            * conc_a
            * fator_pressao
            * fator_cat
        )

    else:
        velocidade = (
            k
            * conc_a
            * conc_b
            * fator_pressao
            * fator_cat
        )

    conversao = 100 * (1 - np.exp(-velocidade * tempo))

    # --------------------------------------------------------
    # Ruído
    # --------------------------------------------------------

    sigma = 1 + 0.4 * d3

    conversao += rng.normal(0, sigma, n)

    conversao = np.clip(conversao, 0, 100)

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    dados = {
        "Temperatura": temperatura,
        "Pressao": pressao,
        "Tempo_residencia": tempo,
        "Concentracao_A": conc_a,
        "Catalisador": catalisador,
    }

    if not reacao_ab:
        dados["Concentracao_B"] = conc_b

    dados["Conversao"] = conversao

    df = pd.DataFrame(dados)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    taxa_missing = 0.02 + 0.01 * d4

    colunas = [c for c in df.columns if c != "Conversao"]

    for coluna in colunas:
        mascara = rng.random(n) < taxa_missing
        df.loc[mascara, coluna] = np.nan

    return df