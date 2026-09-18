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

    n_classes = 4 + int(str(GRR)[-1])

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



def gerar_embeddings_resnet50(X, batch_size=128):
    """
    Gera embeddings de imagens Fashion-MNIST usando uma ResNet50
    pré-treinada no ImageNet, adaptada para entrada grayscale.

    Parâmetros
    ----------
    X : numpy.ndarray
        Imagens no formato (N, 784), com pixels normalizados em [0, 1]
        ou no intervalo [0, 255].

    batch_size : int
        Número de imagens processadas simultaneamente.

    Retorno
    --------
    embeddings : numpy.ndarray
        Embeddings no formato (N, 2048).
    """
    
    import torch
    import numpy as np
    from torchvision.models import resnet50, ResNet50_Weights
    from torchvision import transforms
    from PIL import Image


    # GPU se disponível
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Dispositivo: {device}")

    # --------------------------------------------------
    # ResNet50 pré-treinada
    # --------------------------------------------------

    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)

    # --------------------------------------------------
    # Modifica a primeira camada para 1 canal
    # --------------------------------------------------

    old_conv = model.conv1

    model.conv1 = torch.nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    # Aproveita os pesos RGB pré-treinados
    with torch.no_grad():
        model.conv1.weight[:] = old_conv.weight.mean(
            dim=1,
            keepdim=True
        )

    # Remove a camada de classificação
    model.fc = torch.nn.Identity()

    model = model.to(device)
    model.eval()

    # --------------------------------------------------
    # Pré-processamento
    # --------------------------------------------------

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485],
            std=[0.229]
        )
    ])

    # --------------------------------------------------
    # Converte X para imagens
    # --------------------------------------------------

    X_tensor = np.asarray(X)

    # Normaliza caso X esteja em [0,255]
    if X_tensor.max() > 1:
        X_tensor = X_tensor / 255.0

    imagens = []

    for img in X_tensor:

        img = img.reshape(28, 28)

        img = Image.fromarray(
            (img * 255).astype(np.uint8)
        )

        img = transform(img)

        imagens.append(img)

    imagens = torch.stack(imagens)

    # --------------------------------------------------
    # Geração dos embeddings em batches
    # --------------------------------------------------

    embeddings = []

    with torch.no_grad():

        for i in range(0, len(imagens), batch_size):

            batch = imagens[i:i + batch_size]
            batch = batch.to(device)

            output = model(batch)

            embeddings.append(
                output.cpu()
            )

    # Junta todos os batches
    embeddings = torch.cat(embeddings)

    return embeddings.numpy()