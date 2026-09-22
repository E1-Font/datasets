import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


def criar_dataset_concreto(GRR):
    """
    Importa o Concrete Compressive Strength Dataset da UCI
    e gera um dataset específico para o aluno com base
    nos 4 últimos dígitos do GRR.

    Parâmetros
    ----------
    GRR : str ou int
        GRR do aluno com 8 dígitos.

    Retorna
    -------
    df : pandas.DataFrame
        Dataset com 850 amostras, atributos originais e
        a variável 'Approved'.
    """

    # ============================================================
    # 1. Verificação do GRR
    # ============================================================

    GRR = str(GRR)

    if not GRR.isdigit():
        raise ValueError("O GRR deve conter apenas números.")

    if len(GRR) != 8:
        raise ValueError("O GRR deve possuir exatamente 8 dígitos.")

    # Utiliza somente os 4 últimos dígitos
    seed = int(GRR[-4:])

    # ============================================================
    # 2. Importação do dataset UCI
    # ============================================================

    concrete_compressive_strength = fetch_ucirepo(id=165)

    X = concrete_compressive_strength.data.features.copy()
    y = concrete_compressive_strength.data.targets.copy()

    # ============================================================
    # 3. Identificação da variável resposta
    # ============================================================

    target_name = "Concrete compressive strength"

    # Garante que y seja uma Series
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    # ============================================================
    # 4. Criação do DataFrame completo
    # ============================================================

    df = X.copy()
    df[target_name] = y.values

    # ============================================================
    # 5. Seleção aleatória de 850 amostras
    # ============================================================

    rng = np.random.default_rng(seed)

    indices = rng.choice(
        len(df),
        size=850,
        replace=False
    )

    df = df.iloc[indices].copy()

    # ============================================================
    # 6. Geração do critério de aprovação
    # ============================================================

    # O GRR determina um limiar entre 30 e 50 MPa.
    threshold = 30 + (seed % 2001) / 100

    # Cria a classificação
    df["Approved"] = np.where(
        df[target_name] >= threshold,
        "yes",
        "no"
    )

    # ============================================================
    # 7. Embaralha novamente as 850 amostras
    # ============================================================

    df = df.sample(
        frac=1,
        random_state=seed
    ).reset_index(drop=True)
    
    return df