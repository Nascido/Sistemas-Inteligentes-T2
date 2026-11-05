"""Baixa Fashion-MNIST da fonte oficial e confere os MD5 publicados.

Uso: python scripts/baixar_dados.py
Utiliza somente a biblioteca padrão; pode ser executado de qualquer pasta.
"""

import hashlib
from pathlib import Path
import urllib.request


RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "dados" / "fashion-mnist"
BASE_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
# https://github.com/zalandoresearch/fashion-mnist#get-the-data
ARQUIVOS = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}


def md5(caminho):
    resumo = hashlib.md5()  # Integridade dos dados, conforme a fonte oficial.
    with caminho.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    for nome, esperado in ARQUIVOS.items():
        caminho = DESTINO / nome
        if caminho.exists() and md5(caminho) == esperado:
            print(f"OK (já disponível): {nome}")
            continue
        temporario = caminho.with_suffix(caminho.suffix + ".part")
        print(f"Baixando {nome}...", flush=True)
        try:
            with urllib.request.urlopen(BASE_URL + nome, timeout=120) as resposta:
                with temporario.open("wb") as arquivo:
                    while bloco := resposta.read(1024 * 1024):
                        arquivo.write(bloco)
            if md5(temporario) != esperado:
                raise ValueError(f"Checksum inválido: {nome}")
            temporario.replace(caminho)
        finally:
            temporario.unlink(missing_ok=True)
        print(f"OK (MD5 conferido): {nome}")
    print(f"Dados disponíveis em: {DESTINO}")


if __name__ == "__main__":
    main()
