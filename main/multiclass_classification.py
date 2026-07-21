import numpy as np
import pandas as pd


def gerar_dataset(matricula, n=1500):

    # =========================================================
    # Matrícula
    # =========================================================

    matricula = str(matricula)

    if len(matricula) < 5 or not matricula.isdigit():
        raise ValueError(
            "A matrícula deve conter pelo menos 5 dígitos."
        )

    rng = np.random.default_rng(int(matricula))

    d1 = int(matricula[-1])
    d2 = int(matricula[-2])
    d3 = int(matricula[-3])
    d4 = int(matricula[-4])
    d5 = int(matricula[-5])


    # =========================================================
    # Características específicas do dataset
    # =========================================================

    incluir_fluido = d1 < 5
    incluir_periodo = d2 < 5
    incluir_pressao = d3 < 5


    # =========================================================
    # Intensidade do ruído
    # =========================================================

    fator_ruido = 0.85 + 0.03 * d4


    # =========================================================
    # Distribuição das classes
    # =========================================================

    p_normal = 0.65 + 0.015 * d5

    # Limitamos para evitar desbalanceamento excessivo
    p_normal = min(p_normal, 0.78)

    restante = 1 - p_normal

    probabilidades = [
        p_normal,
        restante * 0.50,
        restante * 0.32,
        restante * 0.18
    ]

    classes = rng.choice(
        [
            "Normal",
            "Incrustacao",
            "Baixa_vazao",
            "Falha_refrigeracao"
        ],
        size=n,
        p=probabilidades
    )


    # =========================================================
    # Variáveis categóricas
    # =========================================================

    nivel_manutencao = rng.choice(
        ["Ruim", "Regular", "Bom", "Excelente"],
        size=n,
        p=[0.10, 0.25, 0.40, 0.25]
    )


    if incluir_fluido:

        fluido_quente = rng.choice(
            ["Agua", "Oleo_termico", "Solucao_aquosa"],
            size=n,
            p=[0.40, 0.35, 0.25]
        )


    if incluir_periodo:

        periodo_operacao = rng.choice(
            ["Novo", "Intermediario", "Proximo_manutencao"],
            size=n,
            p=[0.30, 0.45, 0.25]
        )


    # =========================================================
    # Condições básicas de operação
    # =========================================================

    temperatura_entrada_quente = rng.normal(
        125,
        8 * fator_ruido,
        n
    )

    temperatura_entrada_fria = rng.normal(
        28,
        3 * fator_ruido,
        n
    )

    vazao_quente = rng.normal(
        100,
        8 * fator_ruido,
        n
    )

    vazao_fria = rng.normal(
        115,
        9 * fator_ruido,
        n
    )


    # =========================================================
    # Efeito do fluido, quando disponível
    # =========================================================

    if incluir_fluido:

        efeito_fluido = {
            "Agua": 1.00,
            "Oleo_termico": 0.82,
            "Solucao_aquosa": 0.92
        }

        fator_fluido = np.array(
            [efeito_fluido[x] for x in fluido_quente]
        )

    else:

        fator_fluido = np.ones(n)


    # =========================================================
    # Efeito da manutenção
    # =========================================================

    efeito_manutencao = {
        "Ruim": 0.85,
        "Regular": 0.92,
        "Bom": 1.00,
        "Excelente": 1.04
    }

    fator_manutencao = np.array(
        [efeito_manutencao[x] for x in nivel_manutencao]
    )


    # =========================================================
    # Efeito do período de operação, quando disponível
    # =========================================================

    if incluir_periodo:

        efeito_periodo = {
            "Novo": 1.04,
            "Intermediario": 1.00,
            "Proximo_manutencao": 0.90
        }

        fator_periodo = np.array(
            [efeito_periodo[x] for x in periodo_operacao]
        )

    else:

        fator_periodo = np.ones(n)


    # =========================================================
    # Eficiência térmica efetiva
    # =========================================================

    eficiencia = (
        0.72
        * fator_fluido
        * fator_manutencao
        * fator_periodo
    )


    # =========================================================
    # Modificação conforme o estado operacional
    # =========================================================

    mascara_incrustacao = classes == "Incrustacao"
    mascara_baixa_vazao = classes == "Baixa_vazao"
    mascara_refrigeracao = classes == "Falha_refrigeracao"


    # ---------------------------------------------------------
    # ---------------------------------------------------------

    eficiencia[mascara_incrustacao] *= rng.normal(
        0.68,
        0.05 * fator_ruido,
        mascara_incrustacao.sum()
    )


    # ---------------------------------------------------------
    # ---------------------------------------------------------

    vazao_quente[mascara_baixa_vazao] *= rng.normal(
        0.58,
        0.06 * fator_ruido,
        mascara_baixa_vazao.sum()
    )


    # ---------------------------------------------------------

    # ---------------------------------------------------------

    vazao_fria[mascara_refrigeracao] *= rng.normal(
        0.50,
        0.07 * fator_ruido,
        mascara_refrigeracao.sum()
    )


    # =========================================================
    # Temperaturas de saída
    # =========================================================

    delta_temperatura = (
        temperatura_entrada_quente
        - temperatura_entrada_fria
    )


    temperatura_saida_quente = (
        temperatura_entrada_quente
        - eficiencia * 0.55 * delta_temperatura
        + rng.normal(
            0,
            1.5 * fator_ruido,
            n
        )
    )


    temperatura_saida_fria = (
        temperatura_entrada_fria
        + eficiencia * 0.42 * delta_temperatura
        + rng.normal(
            0,
            1.5 * fator_ruido,
            n
        )
    )



    temperatura_saida_quente[
        mascara_refrigeracao
    ] += rng.normal(
        12,
        2 * fator_ruido,
        mascara_refrigeracao.sum()
    )


    temperatura_saida_fria[
        mascara_refrigeracao
    ] += rng.normal(
        7,
        1.5 * fator_ruido,
        mascara_refrigeracao.sum()
    )


    # =========================================================
    # Pressões
    #
    # Somente são geradas se estiverem disponíveis para
    # aquele aluno.
    # =========================================================

    if incluir_pressao:

        pressao_entrada = rng.normal(
            5.5,
            0.35 * fator_ruido,
            n
        )

        queda_pressao = (
            0.6
            + 0.00004 * vazao_quente**2
            + rng.normal(
                0,
                0.12 * fator_ruido,
                n
            )
        )


        # Incrustação também aumenta a queda de pressão

        queda_pressao[
            mascara_incrustacao
        ] *= rng.normal(
            1.60,
            0.10 * fator_ruido,
            mascara_incrustacao.sum()
        )


        pressao_saida = (
            pressao_entrada
            - queda_pressao
        )


    # =========================================================
    # Construção do DataFrame
    # =========================================================

    dados = {

        "Temperatura_entrada_quente":
            temperatura_entrada_quente,

        "Temperatura_saida_quente":
            temperatura_saida_quente,

        "Temperatura_entrada_fria":
            temperatura_entrada_fria,

        "Temperatura_saida_fria":
            temperatura_saida_fria,

        "Vazao_quente":
            vazao_quente,

        "Vazao_fria":
            vazao_fria,

        "Nivel_manutencao":
            nivel_manutencao
    }


    # =========================================================
    # Variáveis opcionais
    # =========================================================

    if incluir_pressao:

        dados["Pressao_entrada"] = pressao_entrada
        dados["Pressao_saida"] = pressao_saida


    if incluir_fluido:

        dados["Fluido_quente"] = fluido_quente


    if incluir_periodo:

        dados["Periodo_operacao"] = periodo_operacao


    # =========================================================
    # Target
    # =========================================================

    dados["Estado"] = classes


    df = pd.DataFrame(dados)


    return df