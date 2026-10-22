from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd


URL_ENGRAD = (
    "https://zenodo.org/records/12760772/files/"
    "data.h5?download=1"
)

VARIAVEIS_X = [
    "temperature_2m",
    "relative_humidity_2m",
    "cloud_cover",
    "shortwave_radiation",
]

VARIAVEL_Y = "precipitation"


def normalizar_grr(grr):
    grr = str(grr).strip()

    if not grr.isdigit() or len(grr) > 8:
        raise ValueError("O GRR deve possuir 8 dígitos.")

    return grr.zfill(8)


def carregar_dados(
    grr,
    arquivo="data.h5",
    grrs_turma=None
):
    grr = normalizar_grr(grr)
    arquivo = Path(arquivo)

    # Download do dataset
    if not arquivo.exists():
        urlretrieve(URL_ENGRAD, arquivo)

    # Leitura das tabelas do arquivo HDF5
    dados = pd.read_hdf(arquivo, key="data")
    metadata = pd.read_hdf(arquivo, key="metadata")

    # Verificação do esquema real
    metadata_esperado = [
        "city",
        "county",
        "admin_name",
        "lat",
        "lon",
        "elevation",
        "population",
    ]

    if set(metadata.columns) != set(metadata_esperado):
        raise ValueError(
            f"Colunas inesperadas em metadata: "
            f"{metadata.columns.tolist()}"
        )

    metadata = metadata.loc[:, metadata_esperado]

    canais = set(dados.columns.get_level_values(1))
    canais_esperados = set(VARIAVEIS_X + [VARIAVEL_Y])

    if canais != canais_esperados:
        raise ValueError(
            f"Canais inesperados em data: {sorted(canais)}"
        )

    # Lista determinística de cidades
    cidades = (
        metadata
        .dropna(subset=["city"])
        .sort_values(["city", "county", "lat", "lon"])
        .drop_duplicates(subset=["city"])
    )

    if grrs_turma is None:
        # Mapeamento individual pelo valor numérico do GRR
        posicao = int(grr) % len(cidades)

    else:
        # Garante cidades diferentes para os GRRs fornecidos
        grrs_turma = [normalizar_grr(item) for item in grrs_turma]

        if len(grrs_turma) != len(set(grrs_turma)):
            raise ValueError("A lista contém GRRs repetidos.")

        if len(grrs_turma) > len(cidades):
            raise ValueError(
                "Há mais alunos do que cidades disponíveis."
            )

        if grr not in grrs_turma:
            raise ValueError(
                "O GRR informado não está em grrs_turma."
            )

        posicao = sorted(grrs_turma).index(grr)

    # Identificador da localidade no índice de metadata
    node_id = cidades.index[posicao]
    cidade = metadata.loc[node_id, "city"]

    # Seleciona as cinco variáveis da localidade
    serie = dados.xs(
        node_id,
        axis=1,
        level=0
    )

    # X: tempo como coluna + variáveis meteorológicas
    X = serie[VARIAVEIS_X].copy()
    X.index.name = "time"
    X = X.reset_index()

    # y: precipitation
    y = serie[[VARIAVEL_Y]].copy()
    y.index.name = "time"

    # Informações da localidade
    X.attrs["grr"] = grr
    X.attrs["city"] = cidade
    X.attrs["node_id"] = node_id
    X.attrs["metadata"] = metadata.loc[node_id].to_dict()

    return X, y