import os
import requests
import numpy as np

from PIL import Image
from io import BytesIO


def gerar_dataset_pokemon(
    grr,
    n_classes=4,
    imagens_por_classe=30,
    tamanho_imagem=64,
    pasta_cache="pokemon_cache"
):
    """
    Gera um dataset de imagens de Pokémon para PCA + classificação.

    Cada classe corresponde a uma espécie de Pokémon.

    O GRR determina quais Pokémon serão selecionados.

    As imagens são obtidas de diferentes gerações e representações
    disponíveis no repositório de sprites da PokéAPI.

    O fundo das imagens é normalizado para evitar que o PCA aprenda
    diferenças de fundo (preto/branco) em vez das características
    do Pokémon.

    Retorna
    -------
    X_img : ndarray
        Imagens no formato (n_amostras, tamanho_imagem, tamanho_imagem).

    X : ndarray
        Imagens achatadas.

    y : ndarray
        Classes codificadas.

    classes : list
        Nomes dos Pokémon utilizados.
    """

    grr = int(grr)
    rng = np.random.default_rng(grr)

    os.makedirs(
        pasta_cache,
        exist_ok=True
    )

    # =========================================================
    # Primeiros 151 Pokémon
    # =========================================================

    pokemon_disponiveis = np.arange(
        1,
        152
    )

    # Seleção determinada pelo GRR
    ids_selecionados = rng.choice(
        pokemon_disponiveis,
        size=n_classes,
        replace=False
    )

    # =========================================================
    # URLs das diferentes representações
    # =========================================================

    def gerar_urls(pokemon_id):

        urls = []

        # -----------------------------------------------------
        # Sprites padrão
        # -----------------------------------------------------

        urls += [
            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"{pokemon_id}.png",

            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"back/{pokemon_id}.png",

            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"shiny/{pokemon_id}.png",

            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"back/shiny/{pokemon_id}.png"
        ]

        # -----------------------------------------------------
        # Geração I
        # -----------------------------------------------------

        urls += [
            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"versions/generation-i/"
            f"red-blue/{pokemon_id}.png",

            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"versions/generation-i/"
            f"yellow/{pokemon_id}.png"
        ]

        # -----------------------------------------------------
        # Geração II
        # -----------------------------------------------------

        urls += [
            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"versions/generation-ii/"
            f"gold/{pokemon_id}.png",

            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"versions/generation-ii/"
            f"silver/{pokemon_id}.png",

            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"versions/generation-ii/"
            f"crystal/{pokemon_id}.png"
        ]

        # -----------------------------------------------------
        # Geração III
        # -----------------------------------------------------

        urls += [
            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"versions/generation-iii/"
            f"emerald/{pokemon_id}.png",

            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"versions/generation-iii/"
            f"fire-red-leaf-green/{pokemon_id}.png",

            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"versions/generation-iii/"
            f"ruby-sapphire/{pokemon_id}.png"
        ]

        # -----------------------------------------------------
        # Geração IV
        # -----------------------------------------------------

        urls += [
            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"versions/generation-iv/"
            f"diamond-pearl/{pokemon_id}.png",

            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"versions/generation-iv/"
            f"platinum/{pokemon_id}.png",

            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"versions/generation-iv/"
            f"heartgold-soulsilver/{pokemon_id}.png"
        ]

        # -----------------------------------------------------
        # Geração V
        # -----------------------------------------------------

        urls += [
            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"versions/generation-v/"
            f"black-white/{pokemon_id}.png"
        ]

        # -----------------------------------------------------
        # Geração VI
        # -----------------------------------------------------

        urls += [
            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"versions/generation-vi/"
            f"x-y/{pokemon_id}.png",

            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"versions/generation-vi/"
            f"omegaruby-alphasapphire/{pokemon_id}.png"
        ]

        # -----------------------------------------------------
        # Geração VII
        # -----------------------------------------------------

        urls += [
            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"versions/generation-vii/"
            f"ultra-sun-ultra-moon/{pokemon_id}.png"
        ]

        # -----------------------------------------------------
        # Official artwork
        # -----------------------------------------------------

        urls += [
            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"other/official-artwork/"
            f"{pokemon_id}.png",

            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"other/official-artwork/"
            f"shiny/{pokemon_id}.png"
        ]

        # -----------------------------------------------------
        # Home
        # -----------------------------------------------------

        urls += [
            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"other/home/"
            f"{pokemon_id}.png",

            f"https://raw.githubusercontent.com/"
            f"PokeAPI/sprites/master/sprites/pokemon/"
            f"other/home/"
            f"shiny/{pokemon_id}.png"
        ]

        return urls

    # =========================================================
    # Obter nome do Pokémon
    # =========================================================

    def obter_nome(pokemon_id):

        url = (
            "https://pokeapi.co/api/v2/pokemon/"
            + str(pokemon_id)
        )

        try:

            resposta = requests.get(
                url,
                timeout=20
            )

            resposta.raise_for_status()

            return resposta.json()["name"]

        except Exception:

            return f"pokemon_{pokemon_id}"

    # =========================================================
    # Normalização do fundo
    # =========================================================

    def normalizar_fundo(img):

        # Converter para RGBA
        img = img.convert("RGBA")

        arr = np.asarray(
            img,
            dtype=np.float32
        )

        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]

        # -----------------------------------------------------
        # Caso a imagem possua transparência
        # -----------------------------------------------------

        if np.min(alpha) < 250:

            fundo = np.ones(
                rgb.shape,
                dtype=np.float32
            ) * 255

            alpha_norm = (
                alpha[:, :, None] / 255.0
            )

            rgb = (
                rgb * alpha_norm
                +
                fundo * (
                    1 - alpha_norm
                )
            )

        # -----------------------------------------------------
        # Caso seja uma imagem sem transparência
        #
        # Detectamos a cor predominante nas bordas.
        # -----------------------------------------------------

        else:

            bordas = np.concatenate([
                rgb[0, :, :],
                rgb[-1, :, :],
                rgb[:, 0, :],
                rgb[:, -1, :]
            ])

            cor_fundo = np.median(
                bordas,
                axis=0
            )

            distancia = np.linalg.norm(
                rgb - cor_fundo,
                axis=2
            )

            # Pixels suficientemente próximos
            # da cor do fundo
            mascara_fundo = (
                distancia < 25
            )

            rgb[
                mascara_fundo
            ] = 255

        # -----------------------------------------------------
        # Converter para escala de cinza
        # -----------------------------------------------------

        cinza = (
            0.299 * rgb[:, :, 0]
            +
            0.587 * rgb[:, :, 1]
            +
            0.114 * rgb[:, :, 2]
        )

        return cinza.astype(
            np.uint8
        )

    # =========================================================
    # Processamento
    # =========================================================

    X_img = []
    y = []
    classes = []

    for classe_id, pokemon_id in enumerate(
        ids_selecionados
    ):

        nome = obter_nome(
            pokemon_id
        )

        classes.append(
            nome
        )

        print(
            f"\nClasse {classe_id}: "
            f"{nome} (ID {pokemon_id})"
        )

        urls = gerar_urls(
            pokemon_id
        )

        imagens_classe = []

        for j, url in enumerate(urls):

            # Já atingimos o número desejado
            if len(imagens_classe) >= imagens_por_classe:
                break

            try:

                resposta = requests.get(
                    url,
                    timeout=20
                )

                if resposta.status_code != 200:
                    continue

                # Abrir imagem
                img = Image.open(
                    BytesIO(
                        resposta.content
                    )
                )

                # Normalizar fundo
                img = normalizar_fundo(
                    img
                )

                # PIL novamente
                img = Image.fromarray(
                    img
                )

                # Redimensionar
                img = img.resize(
                    (
                        tamanho_imagem,
                        tamanho_imagem
                    ),
                    Image.Resampling.LANCZOS
                )

                # Array
                img = np.asarray(
                    img,
                    dtype=np.float32
                )

                # Normalizar 0–1
                img = img / 255.0

                imagens_classe.append(
                    img
                )

            except Exception:
                continue

        # -----------------------------------------------------
        # Adicionar classe
        # -----------------------------------------------------

        print(
            f"Imagens obtidas: "
            f"{len(imagens_classe)}"
        )

        for img in imagens_classe:

            X_img.append(
                img
            )

            y.append(
                classe_id
            )

    # =========================================================
    # Converter
    # =========================================================

    X_img = np.array(
        X_img,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=int
    )

    # =========================================================
    # Embaralhar
    # =========================================================

    indices = rng.permutation(
        len(y)
    )

    X_img = X_img[
        indices
    ]

    y = y[
        indices
    ]

    # =========================================================
    # Flatten
    # =========================================================

    X = X_img.reshape(
        len(X_img),
        -1
    )

    return (
        X_img,
        X,
        y,
        classes
    )



import numpy as np
import matplotlib.pyplot as plt


def plotar_imagens(X_img, y, classes, n_imagens=5):
    """
    Plota algumas imagens de cada classe.

    Parâmetros
    ----------
    X_img : ndarray
        Imagens no formato (n_amostras, altura, largura).

    y : ndarray
        Classes codificadas.

    classes : list
        Nomes das classes.

    n_imagens : int
        Número de imagens exibidas por classe.
    """

    fig, axes = plt.subplots(
        len(classes),
        n_imagens,
        figsize=(12, 3 * len(classes))
    )

    # Garante que axes seja uma matriz mesmo quando
    # houver apenas uma classe
    axes = np.atleast_2d(axes)

    for classe_id, classe in enumerate(classes):

        indices = np.where(
            y == classe_id
        )[0]

        n = min(
            n_imagens,
            len(indices)
        )

        # Seleciona imagens aleatoriamente
        selecionados = np.random.choice(
            indices,
            size=n,
            replace=False
        )

        for j, indice in enumerate(selecionados):

            axes[classe_id, j].imshow(
                X_img[indice],
                cmap="gray",
                vmin=0,
                vmax=1
            )

            axes[classe_id, j].axis("off")

            if j == 0:
                axes[
                    classe_id,
                    j
                ].set_title(
                    classe.capitalize()
                )

    plt.tight_layout()
    plt.show()