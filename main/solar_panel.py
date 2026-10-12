import gdown
from pathlib import Path
from PIL import Image
import numpy as np
import zipfile
import shutil
import torch


def carregar_dataset(grr, img_size=(128, 128)):

    # ==================================================
    # LINKS DOS ARQUIVOS ZIP NO GOOGLE DRIVE
    # ==================================================

    links = {
        "clean":
            "https://drive.google.com/file/d/1MMtt8cHPbs7sDbK08yUDQWnHUjUUdelS/view?usp=sharing",

        "dusty":
            "https://drive.google.com/file/d/1PokusEHVnjdLrvnbVapq73OuVnCtiLKP/view?usp=sharing",

        "bird_drop":
            "https://drive.google.com/file/d/1Kmgx1BewKEZJvmgD9I-zdSmF4CCFnVjP/view?usp=sharing",

        "electrical_damage":
            "https://drive.google.com/file/d/1_rrecYQOrUqZeTNp5pk5NzbVI-f1UyDa/view?usp=sharing",

        "physical_damage":
            "https://drive.google.com/file/d/1A05aloGt7e7PPvGlVDWzIW-URpnlID-9/view?usp=sharing",

        "snow_covered":
            "https://drive.google.com/file/d/1ImiTsrgHT1_21KMDDJ1RKrLfIsxM8nTD/view?usp=sharing"
    }

 # ==================================================
    # DEFEITOS
    # ==================================================

    defeitos = [
        "dusty",
        "bird_drop",
        "electrical_damage",
        "physical_damage",
        "snow_covered"
    ]

    # ==================================================
    # SELEÇÃO DO DEFEITO PELO GRR
    # ==================================================

    digito = int(str(grr)[-1])

    defeito = defeitos[digito % 5]

    print(f"GRR: {grr}")
    print(f"Defeito selecionado: {defeito}")

    # ==================================================
    # DIRETÓRIO TEMPORÁRIO
    # ==================================================

    pasta_base = Path("/content/solar_panel")

    if pasta_base.exists():
        shutil.rmtree(pasta_base)

    pasta_base.mkdir(parents=True)

    # ==================================================
    # FUNÇÃO PARA BAIXAR E EXTRAIR ZIP
    # ==================================================

    def baixar_zip(nome):

        arquivo_zip = pasta_base / f"{nome}.zip"
        pasta = pasta_base / nome

        print(f"\nBaixando {nome}...")

        gdown.download(
            url=links[nome],
            output=str(arquivo_zip),
            fuzzy=True,
            quiet=False
        )

        if not zipfile.is_zipfile(arquivo_zip):
            raise ValueError(
                f"O arquivo baixado para '{nome}' "
                "não é um ZIP válido. "
                "Verifique o link do Google Drive."
            )

        print(f"Extraindo {nome}...")

        pasta.mkdir(parents=True)

        with zipfile.ZipFile(arquivo_zip, "r") as zip_ref:
            zip_ref.extractall(pasta)

        arquivo_zip.unlink()

        return pasta

    # ==================================================
    # BAIXAR SOMENTE CLEAN + DEFEITO
    # ==================================================

    pasta_clean = baixar_zip("clean")
    pasta_defeito = baixar_zip(defeito)

    # ==================================================
    # CARREGAR IMAGENS
    # ==================================================

    def carregar_imagens(diretorio):

        imagens = []

        extensoes = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        }

        for arquivo in sorted(Path(diretorio).rglob("*")):

            if arquivo.suffix.lower() not in extensoes:
                continue

            try:
                img = Image.open(arquivo).convert("RGB")
                img = img.resize(img_size)

                img = np.asarray(
                    img,
                    dtype=np.float32
                ) / 255.0

                # HWC -> CHW
                img = torch.from_numpy(
                    img
                ).permute(2, 0, 1).contiguous()

                imagens.append(img)

            except Exception as e:
                print(f"Erro ao carregar {arquivo}: {e}")

        if len(imagens) == 0:
            raise ValueError(
                f"Nenhuma imagem encontrada em {diretorio}"
            )

        return torch.stack(imagens)
    
    # ==================================================
    # CARREGAR AS DUAS CLASSES
    # ==================================================

    X_clean = carregar_imagens(pasta_clean)
    X_faulty = carregar_imagens(pasta_defeito)

    print("\nDataset:")
    print(f"  Clean:  {len(X_clean)} imagens")
    print(f"  Faulty: {len(X_faulty)} imagens")

    print(f"\nShape X_clean:  {X_clean.shape}")
    print(f"Shape X_faulty: {X_faulty.shape}")

    # Retorno da função principal
    return X_clean, X_faulty



import matplotlib.pyplot as plt
import numpy as np


import numpy as np
import torch
import matplotlib.pyplot as plt


def plotar_imagens(X_clean, X_faulty, n=5, seed=42):

    rng = np.random.default_rng(seed)

    n_clean = min(n, len(X_clean))
    n_faulty = min(n, len(X_faulty))
    n_colunas = max(n_clean, n_faulty)

    idx_clean = rng.choice(
        len(X_clean),
        n_clean,
        replace=False
    )

    idx_faulty = rng.choice(
        len(X_faulty),
        n_faulty,
        replace=False
    )

    fig, axes = plt.subplots(
        2,
        n_colunas,
        figsize=(3 * n_colunas, 6)
    )

    axes = np.atleast_2d(axes)

    def preparar_imagem(X, idx):

        imagem = X[idx]

        if torch.is_tensor(imagem):
            imagem = imagem.detach().cpu().numpy()

        # CHW -> HWC
        if imagem.shape[0] in (1, 3):
            imagem = np.transpose(
                imagem,
                (1, 2, 0)
            )

        return np.clip(imagem, 0, 1)

    for i, idx in enumerate(idx_clean):

        imagem = preparar_imagem(
            X_clean,
            idx
        )

        axes[0, i].imshow(imagem)
        axes[0, i].set_title("Clean")
        axes[0, i].axis("off")

    for i, idx in enumerate(idx_faulty):

        imagem = preparar_imagem(
            X_faulty,
            idx
        )

        axes[1, i].imshow(imagem)
        axes[1, i].set_title("Faulty")
        axes[1, i].axis("off")

    for i in range(n_clean, n_colunas):
        axes[0, i].axis("off")

    for i in range(n_faulty, n_colunas):
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()