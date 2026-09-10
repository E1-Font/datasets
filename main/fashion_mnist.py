import hashlib
import numpy as np
from tensorflow.keras.datasets import fashion_mnist


def gerar_dataset_fashion_mnist(
    GRR,
    n_classes=6,
    n_por_classe=90,
    normalizar=True
):
    """
    Gera um dataset individual de Fashion-MNIST determinado pelo GRR.

    Parâmetros
    ----------
    GRR : str ou int
        GRR do aluno. Determina a seleção das classes e das imagens.

    n_classes : int, default=6
        Número de classes utilizadas no dataset.

    n_por_classe : int, default=30
        Número de imagens selecionadas por classe.

    normalizar : bool, default=True
        Se True, os pixels são normalizados para o intervalo [0, 1].

    Retorno
    -------
    X : numpy.ndarray
        Dataset contendo as imagens, sem os labels.
        Shape: (n_classes * n_por_classe, 784)

    y : numpy.ndarray
        Labels correspondentes às imagens.
        Shape: (n_classes * n_por_classe,)

    classes : numpy.ndarray
        Classes do Fashion-MNIST presentes no dataset.

    """

    GRR = str(GRR)

    n_classes = 4 + GRR[-1]


    # ---------------------------------------------------------
    # 1. Validação
    # ---------------------------------------------------------

    if not 3 <= n_classes <= 10:
        raise ValueError("n_classes deve estar entre 2 e 10.")

    if n_por_classe < 1:
        raise ValueError("n_por_classe deve ser maior que zero.")

    # ---------------------------------------------------------
    # 2. Criar gerador aleatório a partir do GRR
    # ---------------------------------------------------------

    hash_grr = hashlib.sha256(
        GRR.encode("utf-8")
    ).hexdigest()

    seed = int(hash_grr[:8], 16)

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # 3. Carregar Fashion-MNIST
    # ---------------------------------------------------------

    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)

    # ---------------------------------------------------------
    # 4. Selecionar classes
    # ---------------------------------------------------------

    classes = rng.choice(
        np.arange(10),
        size=n_classes,
        replace=False
    )

    # ---------------------------------------------------------
    # 5. Selecionar imagens
    # ---------------------------------------------------------

    indices = []

    for classe in classes:

        indices_classe = np.where(y == classe)[0]

        indices_classe = rng.choice(
            indices_classe,
            size=n_por_classe,
            replace=False
        )

        indices.extend(indices_classe)

    indices = np.array(indices)

    # ---------------------------------------------------------
    # 6. Embaralhar as amostras
    # ---------------------------------------------------------

    rng.shuffle(indices)

    X = X[indices]
    y = y[indices]

    # ---------------------------------------------------------
    # 7. Normalizar
    # ---------------------------------------------------------

    X = X.astype(np.float32)

    if normalizar:
        X /= 255.0

    # ---------------------------------------------------------
    # 8. Transformar 28x28 em vetor de 784 atributos
    # ---------------------------------------------------------

    X = X.reshape(X.shape[0], -1)

    return X, y, classes


import numpy as np
import matplotlib.pyplot as plt


def plotar_imagens(X, n_imagens=10, seed=None):
    """
    Plota imagens aleatórias de um dataset Fashion-MNIST.

    Parâmetros
    ----------
    X : numpy.ndarray
        Dataset contendo as imagens.
        Pode ter shape (N, 784) ou (N, 28, 28).

    n_imagens : int, default=10
        Número de imagens a serem exibidas.

    seed : int ou None, default=None
        Semente para tornar a seleção das imagens reprodutível.
        Se None, as imagens serão selecionadas aleatoriamente
        a cada execução.

    """

    if len(X) == 0:
        raise ValueError("O dataset X está vazio.")

    if n_imagens < 1:
        raise ValueError("n_imagens deve ser maior que zero.")

    n_imagens = min(n_imagens, len(X))

    # Gerador aleatório
    rng = np.random.default_rng(seed)

    # Selecionar imagens aleatoriamente
    indices = rng.choice(
        len(X),
        size=n_imagens,
        replace=False
    )

    # Configuração da figura
    n_colunas = min(5, n_imagens)
    n_linhas = int(np.ceil(n_imagens / n_colunas))

    fig, axes = plt.subplots(
        n_linhas,
        n_colunas,
        figsize=(2 * n_colunas, 2 * n_linhas)
    )

    # Garantir que axes seja iterável
    axes = np.atleast_1d(axes).flatten()

    for ax, indice in zip(axes, indices):

        imagem = X[indice]

        # Caso a imagem esteja armazenada como vetor
        if imagem.size == 784:
            imagem = imagem.reshape(28, 28)

        ax.imshow(imagem, cmap="gray")
        ax.set_title(f"Amostra {indice}")
        ax.axis("off")

    # Remover eixos vazios
    for ax in axes[n_imagens:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()