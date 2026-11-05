# Dados

Fashion-MNIST: imagens 28 × 28 em escala de cinza, com 10 classes de vestuário.
A divisão oficial contém 60.000 exemplos de treino e 10.000 de teste.

Fonte e checksums: [Zalando Research — Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist#get-the-data).
Referência: Han Xiao, Kashif Rasul e Roland Vollgraf (2017),
[Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms](https://arxiv.org/abs/1708.07747).
Licença: MIT; consulte `LICENSE-Fashion-MNIST.txt`.

Para baixar ou verificar os quatro arquivos originais:

```bash
python scripts/baixar_dados.py
```

O script mantém os arquivos compactados em `dados/fashion-mnist/`, verifica os
MD5 publicados e reaproveita arquivos válidos. Os dados não são versionados.
O notebook também executa essa verificação antes do carregamento; se os arquivos
já estiverem disponíveis e íntegros, não é necessária conexão à internet.

O notebook reserva 20% do treino oficial para validação estratificada, com seed 42
(48.000 para treino e 12.000 para validação). O teste permanece reservado para a
avaliação final, após escolher a CNN pelos resultados de validação.
