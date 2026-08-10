import numpy as np
import pandas as pd


def gerar_dataset(grr):
    """
    Gera um pequeno dataset para regressao e classificacao
    aplicado ao escoamento em um leito particulado.

    Variavel alvo de regressao:
        Queda_pressao [Pa]

    Variavel alvo de classificacao:
        Regime
            0 = Regime viscoso
            1 = Regime inercial

    O GRR controla:
        - tamanho da amostra;
        - variaveis disponiveis;
        - nivel de ruido;
        - faixa de velocidades;
        - seed aleatoria.
    """

    # =========================================================
    # Validacao do GRR
    # =========================================================

    grr = str(grr)

    if len(grr) < 4 or not grr.isdigit():
        raise ValueError(
            "O GRR deve possuir pelo menos quatro digitos."
        )

    rng = np.random.default_rng(int(grr))

    # =========================================================
    # Digitos do GRR
    # =========================================================

    d1 = int(grr[-1])
    d2 = int(grr[-2])
    d3 = int(grr[-3])
    d4 = int(grr[-4])

    # =========================================================
    # Tamanho da amostra
    # =========================================================

    if d3 <= 3:
        n = 80
    elif d3 <= 6:
        n = 100
    else:
        n = 120

    # =========================================================
    # Variaveis disponiveis
    # =========================================================

    incluir_densidade = d1 >= 3
    incluir_rugosidade = d1 >= 7

    # =========================================================
    # Nivel de ruido
    # =========================================================

    if d2 <= 3:
        ruido_relativo = 0.02
    elif d2 <= 6:
        ruido_relativo = 0.05
    else:
        ruido_relativo = 0.08

    # =========================================================
    # Velocidade superficial
    #
    # GRR 0-4:
    #     velocidades predominantemente menores
    #
    # GRR 5-9:
    #     velocidades predominantemente maiores
    # =========================================================

    if d4 <= 4:
        velocidade = rng.uniform(
            0.01,
            0.15,
            n
        )
    else:
        velocidade = rng.uniform(
            0.05,
            0.80,
            n
        )

    # =========================================================
    # Variaveis fisicas
    # =========================================================

    diametro_particula = rng.uniform(
        0.5e-3,
        5.0e-3,
        n
    )

    porosidade = rng.uniform(
        0.35,
        0.50,
        n
    )

    viscosidade = rng.uniform(
        0.0008,
        0.0030,
        n
    )

    comprimento_leito = rng.uniform(
        0.20,
        1.50,
        n
    )

    # =========================================================
    # Densidade do fluido
    # =========================================================

    if incluir_densidade:

        densidade_fluido = rng.uniform(
            850,
            1200,
            n
        )

    else:

        # Valor fixo utilizado somente para a geracao da
        # resposta e da classificacao.
        densidade_fluido = np.full(
            n,
            1000.0
        )

    # =========================================================
    # Rugosidade
    # =========================================================

    if incluir_rugosidade:

        rugosidade = rng.uniform(
            0.01e-3,
            0.15e-3,
            n
        )

        fator_rugosidade = (
            1.0
            + 1.5
            * rugosidade
            / diametro_particula
        )

    else:

        rugosidade = None

        fator_rugosidade = np.ones(n)

    # =========================================================
    # NUMERO DE REYNOLDS DA PARTICULA
    # =========================================================

    numero_reynolds = (
        densidade_fluido
        * velocidade
        * diametro_particula
        / viscosidade
    )

    # =========================================================
    # VARIAVEL DE CLASSIFICACAO
    #
    # O limiar e definido pela mediana do proprio dataset.
    #
    # 0 -> Regime viscoso
    # 1 -> Regime inercial
    #
    # Dessa forma, ambos os grupos possuem observacoes.
    # =========================================================

    limiar_reynolds = np.median(
        numero_reynolds
    )

    regime = (
        numero_reynolds >= limiar_reynolds
    ).astype(int)

    # =========================================================
    # EQUACAO DE ERGUN
    # =========================================================

    termo_viscoso = (

        150
        * viscosidade
        * (1 - porosidade) ** 2

        / (
            diametro_particula ** 2
            * porosidade ** 3
        )

        * velocidade
    )

    termo_inercial = (

        1.75
        * densidade_fluido
        * (1 - porosidade)

        / (
            diametro_particula
            * porosidade ** 3
        )

        * velocidade ** 2
    )

    queda_pressao = (

        comprimento_leito

        * (
            termo_viscoso
            + fator_rugosidade
            * termo_inercial
        )
    )

    # =========================================================
    # Pequena nao-linearidade experimental
    # =========================================================

    fator_nao_linear = (

        1

        + 0.025
        * np.sin(6 * velocidade)

        + 0.015
        * (porosidade - 0.425)
    )

    queda_pressao *= fator_nao_linear

    # =========================================================
    # Ruido experimental
    # =========================================================

    queda_pressao *= (

        1

        + rng.normal(
            0,
            ruido_relativo,
            n
        )
    )

    # Evita valores negativos

    queda_pressao = np.maximum(
        queda_pressao,
        0
    )

    # =========================================================
    # CONSTRUCAO DO DATAFRAME
    # =========================================================

    dados = {

        "Velocidade_superficial":
            velocidade,

        "Diametro_particula":
            diametro_particula,

        "Porosidade":
            porosidade,

        "Viscosidade":
            viscosidade,

        "Comprimento_leito":
            comprimento_leito
    }

    # =========================================================
    # Variavel opcional: densidade
    # =========================================================

    if incluir_densidade:

        dados["Densidade_fluido"] = (
            densidade_fluido
        )

    # =========================================================
    # Variavel opcional: rugosidade
    # =========================================================

    if incluir_rugosidade:

        dados["Rugosidade"] = (
            rugosidade
        )

    # =========================================================
    # VARIAVEIS ALVO
    # =========================================================

    dados["Queda_pressao"] = (
        queda_pressao
    )

    dados["Regime"] = (
        regime
    )

    # =========================================================
    # DataFrame final
    # =========================================================

    df = pd.DataFrame(dados)

    return df