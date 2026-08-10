import numpy as np
import pandas as pd


def gerar_dataset(grr):
    """Gera um pequeno dataset para regressao da queda de pressao
    em um leito particulado. O GRR controla tamanho, variaveis,
    ruido e faixa de velocidades.
    """
    grr = str(grr)
    if len(grr) < 4 or not grr.isdigit():
        raise ValueError("O GRR deve possuir pelo menos quatro digitos.")

    rng = np.random.default_rng(int(grr))

    d1 = int(grr[-1])
    d2 = int(grr[-2])
    d3 = int(grr[-3])
    d4 = int(grr[-4])

    # Tamanho: 80, 100 ou 120 observacoes
    n = 80 if d3 <= 3 else (100 if d3 <= 6 else 120)

    # Variaveis opcionais
    incluir_densidade = d1 >= 3
    incluir_rugosidade = d1 >= 7

    # Ruido: 2%, 5% ou 8%
    ruido_relativo = 0.02 if d2 <= 3 else (0.05 if d2 <= 6 else 0.08)

    # Faixa de velocidade: baixa ou alta
    if d4 <= 4:
        velocidade = rng.uniform(0.01, 0.15, n)
    else:
        velocidade = rng.uniform(0.05, 0.80, n)

    # Variaveis fisicas
    diametro_particula = rng.uniform(0.5e-3, 5.0e-3, n)
    porosidade = rng.uniform(0.35, 0.50, n)
    viscosidade = rng.uniform(0.0008, 0.0030, n)
    comprimento_leito = rng.uniform(0.20, 1.50, n)

    # Se a densidade nao estiver disponivel, usa-se uma densidade
    # de referencia fixa na geracao. Portanto, nao ha variavel latente.
    if incluir_densidade:
        densidade_fluido = rng.uniform(850, 1200, n)
    else:
        densidade_fluido = np.full(n, 1000.0)

    # Rugosidade, quando disponivel
    if incluir_rugosidade:
        rugosidade = rng.uniform(0.01e-3, 0.15e-3, n)
        fator_rugosidade = 1.0 + 1.5 * rugosidade / diametro_particula
    else:
        rugosidade = None
        fator_rugosidade = np.ones(n)

    # Equacao de Ergun
    termo_viscoso = (
        150 * viscosidade * (1 - porosidade) ** 2
        / (diametro_particula ** 2 * porosidade ** 3)
        * velocidade
    )

    termo_inercial = (
        1.75 * densidade_fluido * (1 - porosidade)
        / (diametro_particula * porosidade ** 3)
        * velocidade ** 2
    )

    queda_pressao = comprimento_leito * (
        termo_viscoso + fator_rugosidade * termo_inercial
    )

    # Pequenas nao idealidades experimentais
    fator_nao_linear = (
        1 + 0.025 * np.sin(6 * velocidade)
        + 0.015 * (porosidade - 0.425)
    )
    queda_pressao *= fator_nao_linear

    # Ruido experimental proporcional a resposta
    queda_pressao *= (1 + rng.normal(0, ruido_relativo, n))
    queda_pressao = np.maximum(queda_pressao, 0)

    # DataFrame
    dados = {
        "Velocidade_superficial": velocidade,
        "Diametro_particula": diametro_particula,
        "Porosidade": porosidade,
        "Viscosidade": viscosidade,
        "Comprimento_leito": comprimento_leito,
    }

    if incluir_densidade:
        dados["Densidade_fluido"] = densidade_fluido

    if incluir_rugosidade:
        dados["Rugosidade"] = rugosidade

    dados["Queda_pressao"] = queda_pressao

    return pd.DataFrame(dados)