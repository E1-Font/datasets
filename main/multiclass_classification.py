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
    # Variável categórica do aluno
    #
    # Cada dataset possui EXATAMENTE uma variável categórica.
    # =========================================================

    if d1 <= 3:
        variavel_categorica = "Nivel_manutencao"

    elif d1 <= 6:
        variavel_categorica = "Fluido_quente"

    else:
        variavel_categorica = "Periodo_operacao"


    # =========================================================
    # Presença das variáveis de pressão
    # =========================================================

    incluir_pressao = d2 < 5


    # =========================================================
    # Intensidade do ruído
    #
    # Varia aproximadamente entre 0.85 e 1.12
    # =========================================================

    fator_ruido = 0.85 + 0.03 * d3


    # =========================================================
    # Distribuição das classes
    #
    # O grau de desbalanceamento depende da matrícula.
    # =========================================================

    p_normal = 0.65 + 0.015 * d4

    p_normal = min(p_normal, 0.78)

    restante = 1.0 - p_normal


    # O último dígito também altera ligeiramente a distribuição
    # das classes minoritárias.

    fracao_incrustacao = 0.46 + 0.01 * d5

    fracao_baixa_vazao = 0.34 - 0.005 * d5

    fracao_refrigeracao = (
        1.0
        - fracao_incrustacao
        - fracao_baixa_vazao
    )


    probabilidades = [

        p_normal,

        restante * fracao_incrustacao,

        restante * fracao_baixa_vazao,

        restante * fracao_refrigeracao
    ]


    # Garante que a soma seja exatamente 1
    probabilidades = np.array(probabilidades)

    probabilidades = (
        probabilidades
        / probabilidades.sum()
    )


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
    # Variável categórica
    # =========================================================

    if variavel_categorica == "Nivel_manutencao":

        categoria = rng.choice(

            [
                "Ruim",
                "Regular",
                "Bom",
                "Excelente"
            ],

            size=n,

            p=[
                0.10,
                0.25,
                0.40,
                0.25
            ]
        )


    elif variavel_categorica == "Fluido_quente":

        categoria = rng.choice(

            [
                "Agua",
                "Oleo_termico",
                "Solucao_aquosa"
            ],

            size=n,

            p=[
                0.40,
                0.35,
                0.25
            ]
        )


    else:

        categoria = rng.choice(

            [
                "Novo",
                "Intermediario",
                "Proximo_manutencao"
            ],

            size=n,

            p=[
                0.30,
                0.45,
                0.25
            ]
        )


    # =========================================================
    # Condições básicas de operação
    # =========================================================

    temperatura_entrada_quente = rng.normal(
        loc=125,
        scale=8 * fator_ruido,
        size=n
    )


    temperatura_entrada_fria = rng.normal(
        loc=28,
        scale=3 * fator_ruido,
        size=n
    )


    vazao_quente = rng.normal(
        loc=100,
        scale=8 * fator_ruido,
        size=n
    )


    vazao_fria = rng.normal(
        loc=115,
        scale=9 * fator_ruido,
        size=n
    )


    # =========================================================
    # Fator associado à variável categórica
    #
    # Somente a variável presente no dataset influencia
    # o processo.
    # =========================================================

    if variavel_categorica == "Nivel_manutencao":

        efeito_categoria = {

            "Ruim": 0.85,

            "Regular": 0.92,

            "Bom": 1.00,

            "Excelente": 1.04
        }


    elif variavel_categorica == "Fluido_quente":

        efeito_categoria = {

            "Agua": 1.00,

            "Oleo_termico": 0.82,

            "Solucao_aquosa": 0.92
        }


    else:

        efeito_categoria = {

            "Novo": 1.04,

            "Intermediario": 1.00,

            "Proximo_manutencao": 0.90
        }


    fator_categoria = np.array(

        [
            efeito_categoria[x]
            for x in categoria
        ]
    )


    # =========================================================
    # Eficiência térmica efetiva
    # =========================================================

    eficiencia = (
        0.72
        * fator_categoria
    )


    # =========================================================
    # Máscaras correspondentes às classes
    # =========================================================

    mascara_incrustacao = (
        classes == "Incrustacao"
    )

    mascara_baixa_vazao = (
        classes == "Baixa_vazao"
    )

    mascara_refrigeracao = (
        classes == "Falha_refrigeracao"
    )


    # =========================================================
    # INCRUSTAÇÃO
    #
    # Reduz a eficiência de transferência térmica.
    # =========================================================

    eficiencia[
        mascara_incrustacao
    ] *= rng.normal(

        loc=0.68,

        scale=0.05 * fator_ruido,

        size=mascara_incrustacao.sum()
    )


    # =========================================================
    # BAIXA VAZÃO
    #
    # Redução da vazão da corrente quente.
    # =========================================================

    vazao_quente[
        mascara_baixa_vazao
    ] *= rng.normal(

        loc=0.58,

        scale=0.06 * fator_ruido,

        size=mascara_baixa_vazao.sum()
    )


    # =========================================================
    # FALHA DE REFRIGERAÇÃO
    #
    # Redução da vazão da corrente fria.
    # =========================================================

    vazao_fria[
        mascara_refrigeracao
    ] *= rng.normal(

        loc=0.50,

        scale=0.07 * fator_ruido,

        size=mascara_refrigeracao.sum()
    )


    # =========================================================
    # Diferença de temperatura
    # =========================================================

    delta_temperatura = (

        temperatura_entrada_quente

        - temperatura_entrada_fria
    )


    # =========================================================
    # Temperatura de saída da corrente quente
    # =========================================================

    temperatura_saida_quente = (

        temperatura_entrada_quente

        - eficiencia
        * 0.55
        * delta_temperatura

        + rng.normal(
            loc=0,
            scale=1.5 * fator_ruido,
            size=n
        )
    )


    # =========================================================
    # Temperatura de saída da corrente fria
    # =========================================================

    temperatura_saida_fria = (

        temperatura_entrada_fria

        + eficiencia
        * 0.42
        * delta_temperatura

        + rng.normal(
            loc=0,
            scale=1.5 * fator_ruido,
            size=n
        )
    )


    # =========================================================
    # Efeito adicional da falha de refrigeração
    # =========================================================

    temperatura_saida_quente[
        mascara_refrigeracao
    ] += rng.normal(

        loc=12,

        scale=2 * fator_ruido,

        size=mascara_refrigeracao.sum()
    )


    temperatura_saida_fria[
        mascara_refrigeracao
    ] += rng.normal(

        loc=7,

        scale=1.5 * fator_ruido,

        size=mascara_refrigeracao.sum()
    )


    # =========================================================
    # Construção inicial do DataFrame
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
            vazao_fria
    }


    # =========================================================
    # Pressões
    #
    # Presentes somente para algumas matrículas.
    # =========================================================

    if incluir_pressao:

        pressao_entrada = rng.normal(

            loc=5.5,

            scale=0.35 * fator_ruido,

            size=n
        )


        queda_pressao = (

            0.6

            + 0.00004
            * vazao_quente**2

            + rng.normal(

                loc=0,

                scale=0.12 * fator_ruido,

                size=n
            )
        )


        # A incrustação aumenta a queda de pressão

        queda_pressao[
            mascara_incrustacao
        ] *= rng.normal(

            loc=1.60,

            scale=0.10 * fator_ruido,

            size=mascara_incrustacao.sum()
        )


        pressao_saida = (

            pressao_entrada

            - queda_pressao
        )


        dados["Pressao_entrada"] = (
            pressao_entrada
        )

        dados["Pressao_saida"] = (
            pressao_saida
        )


    # =========================================================
    # Adiciona EXATAMENTE uma variável categórica
    # =========================================================

    dados[variavel_categorica] = categoria


    # =========================================================
    # Variável alvo
    # =========================================================

    dados["Estado"] = classes


    # =========================================================
    # DataFrame
    # =========================================================

    df = pd.DataFrame(dados)


    return df