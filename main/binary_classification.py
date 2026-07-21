import numpy as np
import pandas as pd


def gerar_dataset(matricula, n=1000):

    # =========================================================
    # Matrícula
    # =========================================================

    matricula = str(matricula)

    if len(matricula) < 2 or not matricula.isdigit():
        raise ValueError(
            "A matrícula deve conter pelo menos 2 dígitos."
        )

    # Matrícula utilizada como seed
    rng = np.random.default_rng(int(matricula))

    # Últimos dois dígitos
    d1 = int(matricula[-1])
    d2 = int(matricula[-2])


    # =========================================================
    # Determinação das variáveis categóricas disponíveis
    # =========================================================

    incluir_fluido = d1 < 5
    incluir_turno = d2 < 5


    # =========================================================
    # Variável categórica ordinal
    # Presente para todos os alunos
    # =========================================================

    nivel_manutencao = rng.choice(
        ["Ruim", "Regular", "Bom", "Excelente"],
        size=n,
        p=[0.10, 0.25, 0.40, 0.25]
    )


    # =========================================================
    # Variáveis numéricas básicas
    # =========================================================

    vazao = rng.normal(
        loc=100,
        scale=18,
        size=n
    )

    vazao = np.clip(
        vazao,
        40,
        160
    )


    rotacao = rng.normal(
        loc=1750,
        scale=120,
        size=n
    )

    rotacao = np.clip(
        rotacao,
        1400,
        2100
    )


    temperatura = rng.normal(
        loc=55,
        scale=10,
        size=n
    )

    temperatura = np.clip(
        temperatura,
        25,
        90
    )


    # =========================================================
    # Variáveis categóricas opcionais
    # =========================================================

    if incluir_fluido:

        fluido = rng.choice(
            ["Agua", "Oleo", "Solucao_glicol"],
            size=n,
            p=[0.50, 0.30, 0.20]
        )


    if incluir_turno:

        turno = rng.choice(
            ["Manha", "Tarde", "Noite"],
            size=n,
            p=[0.40, 0.35, 0.25]
        )


    # =========================================================
    # Pressão
    #
    # Relação aproximada com vazão e rotação
    # =========================================================

    pressao = (
        5.0
        + 0.004 * (rotacao - 1750)
        - 0.015 * (vazao - 100)
        + rng.normal(0, 0.35, n)
    )


    # =========================================================
    # Potência
    # =========================================================

    potencia = (
        18
        + 0.09 * vazao
        + 0.006 * (rotacao - 1750)
        + rng.normal(0, 1.2, n)
    )


    # Se a variável Fluido estiver disponível,
    # ela também influencia a potência.

    if incluir_fluido:

        efeito_fluido_potencia = {
            "Agua": 0.0,
            "Oleo": 2.5,
            "Solucao_glicol": 1.2
        }

        potencia += np.array(
            [efeito_fluido_potencia[x] for x in fluido]
        )


    # =========================================================
    # Vibração
    #
    # A vibração aumenta quando a vazão se afasta
    # da região normal de operação.
    # =========================================================

    vibracao = (
        1.5
        + 0.0008 * (vazao - 100)**2
        + rng.normal(0, 0.5, n)
    )

    vibracao = np.clip(
        vibracao,
        0.2,
        None
    )


    # =========================================================
    # Efeito do nível de manutenção
    # Variável ORDINAL
    # =========================================================

    efeito_manutencao = {
        "Ruim": 1.0,
        "Regular": 0.5,
        "Bom": 0.0,
        "Excelente": -0.4
    }

    risco_manutencao = np.array(
        [efeito_manutencao[x] for x in nivel_manutencao]
    )


    # =========================================================
    # Risco básico
    # =========================================================

    risco = (
        -2.0
        + 0.9 * (vibracao - 2.0)
        + 0.045 * (temperatura - 55)
        + 0.035 * np.abs(vazao - 100)
        + risco_manutencao
    )


    # =========================================================
    # Influência do fluido
    #
    # Somente existe se Fluido estiver no dataset
    # =========================================================

    if incluir_fluido:

        efeito_fluido_risco = {
            "Agua": 0.0,
            "Oleo": 0.50,
            "Solucao_glicol": 0.20
        }

        risco += np.array(
            [efeito_fluido_risco[x] for x in fluido]
        )


    # =========================================================
    # Influência do turno
    #
    # Somente existe se Turno estiver no dataset
    # =========================================================

    if incluir_turno:

        efeito_turno = {
            "Manha": 0.0,
            "Tarde": 0.10,
            "Noite": 0.40
        }

        risco += np.array(
            [efeito_turno[x] for x in turno]
        )


    # =========================================================
    # Conversão do risco em probabilidade
    # =========================================================

    probabilidade_falha = (
        1 / (1 + np.exp(-risco))
    )


    # =========================================================
    # Sorteio da classe
    #
    # 0 = Normal
    # 1 = Falha
    # =========================================================

    falha = rng.binomial(
        1,
        probabilidade_falha
    )


    # =========================================================
    # Construção do DataFrame
    # =========================================================

    dados = {

        "Vazao": vazao,
        "Pressao": pressao,
        "Temperatura": temperatura,
        "Rotacao": rotacao,
        "Vibracao": vibracao,
        "Potencia": potencia,

        # Categórica ordinal
        "Nivel_manutencao": nivel_manutencao

    }


    # =========================================================
    # Adiciona somente as categóricas correspondentes
    # à matrícula
    # =========================================================

    if incluir_fluido:
        dados["Fluido"] = fluido

    if incluir_turno:
        dados["Turno"] = turno


    # Target
    dados["Falha"] = falha


    df = pd.DataFrame(dados)


    return df