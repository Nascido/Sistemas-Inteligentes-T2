# Item 2 — CNN com aumento de dados

> **Sobre os resultados:** os valores e análises numéricas deste guia registram
> a execução inicial usada para documentar o código. O notebook da equipe foi
> executado novamente na finalização; consulte suas saídas e análises para os
> resultados da execução final. A semente não garante valores idênticos entre execuções.

Este documento acompanha a seção 5 do
[notebook da equipe](../Trabalho_02_Redes_Neurais.ipynb). Ele explica a construção,
o treinamento e a análise de uma CNN para classificar as dez classes de roupas do
Fashion-MNIST com aumento de dados.

A preparação do dataset e os conceitos de perda e acurácia já são detalhados na
[documentação do item 1](ITEM_1_MLP.md). Aqui, o foco é o que muda na CNN e como
interpretar sua comparação com a MLP.

O documento está organizado em: requisitos e referência; dados; aumento de dados;
construção e compilação; treinamento; leitura das métricas; gráficos e comparação;
resultados; arquivos gerados; referências.

## 1. Requisitos e referência da aula

O [enunciado](<Trabalho Prático 2 - Redes Neurais.pdf>) exige treinar uma CNN com
aumento de dados, explicar a finalidade dessa técnica, apresentar perda e
acurácia de treino e validação por época e analisar o desempenho em comparação
com a MLP, discutindo sobreajuste e subajuste.

A implementação segue a seção **CNN com aumento de dados** da
[Aula 21 — CNN](../Aulas_Praticas/Aula_21_CNN_MNIST_ESTUDANTE.ipynb), de autoria
da **Profa. Nathalia da Cruz Alves**. Foram utilizados como base:

- `data_augmentation`, com rotação, translação e zoom aleatórios;
- a visualização das imagens transformadas;
- `model3`, com três convoluções, pooling e duas camadas densas;
- `history3` e seus gráficos de treino e validação.

Mantivemos a arquitetura do `model3`. Não reproduzimos os outros treinamentos ou
atividades extras da aula, porque o item 2 pede uma CNN com aumento de dados.

| Elemento | Exemplo da aula | Adaptação no trabalho |
| --- | --- | --- |
| Dataset | MNIST de dígitos | Fashion-MNIST fornecido, com dez classes de roupas. |
| Formato | 28 × 28 × 1 | Mesmo formato, aproveitando a preparação existente. |
| Rotação | `RandomRotation(0.1)` | `RandomRotation(0.05)`, para limitar a mudança de orientação das roupas. |
| Translação e zoom | Fatores 0,1 | Mantidos. |
| Validação | `validation_split=0.2` | `validation_data=(X_val, y_val)`, preservando a divisão da MLP. |
| Épocas | 10 | 20, mesma quantidade da MLP. |
| Lote | Não explicitado na chamada da aula | 32, explicitado para documentar o experimento. |
| Adam | Informado como `'adam'` | Instância de Adam com taxa 0,001 explicitada. |
| Gráficos | Curvas de treino e validação | Mantidas, com épocas iniciando em 1 e exportação da figura. |

Aumentar as épocas para 20 não garante uma CNN melhor. É uma escolha de orçamento
para comparar as configurações com a mesma quantidade de passagens pelo treino.
A busca de hiperparâmetros será desenvolvida no item 3.

## 2. Dados utilizados e separação dos conjuntos

O experimento recebe as variáveis preparadas antes da MLP:

| Variável | Formato | Uso |
| --- | --- | --- |
| `X_train` | `(48000, 28, 28, 1)` | Imagens normalizadas para aprender os pesos. |
| `y_train` | `(48000,)` | Rótulos inteiros das imagens de treino. |
| `X_val` | `(12000, 28, 28, 1)` | Imagens normalizadas para acompanhar a generalização. |
| `y_val` | `(12000,)` | Rótulos inteiros das imagens de validação. |

A divisão é estratificada, com semente 42. Cada classe tem 4.800 imagens no treino
e 1.200 na validação. Os mesmos índices utilizados pela MLP são utilizados pela
CNN; não é criada uma nova validação dentro de `fit`.

Os pixels já estão em `float32`, com valores entre 0 e 1. O último eixo representa
o canal de cinza. Não dividimos os valores por 255 novamente.

O teste oficial, com 10.000 imagens, permanece reservado para a avaliação final
do melhor modelo encontrado depois da otimização. Ele não participa deste item.

## 3. Por que utilizar aumento de dados

O aumento de dados produz versões transformadas das imagens de treino. Uma roupa
pode aparecer um pouco deslocada, rotacionada ou com outra escala, sem mudar de
classe. Expor o modelo a essas variações pode reduzir sua dependência de posições
ou apresentações específicas das imagens de treino.

O objetivo é favorecer a generalização. Entretanto, a técnica não garante melhora
em toda configuração. Se as transformações deformarem demais a imagem ou
eliminarem características importantes, podem dificultar o aprendizado.

Nesta implementação, as versões são geradas durante o processamento dos lotes.
Não criamos uma pasta com novas imagens nem aumentamos permanentemente o número
de linhas de `X_train`. Cada época continua utilizando as mesmas 48.000 imagens
de origem, com transformações aleatórias no caminho de treinamento.

### 3.1. Rotação

`RandomRotation(0.05)` utiliza um fator correspondente a uma fração de uma volta
completa. Assim, o limite angular é:

```text
0,05 × 360 graus = 18 graus
```

A camada gera pequenas rotações positivas ou negativas, de até aproximadamente
±18°. O argumento não significa 0,05 grau nem 5 graus.

O fator foi reduzido em relação ao 0,1 usado na aula, equivalente a ±36°, para
limitar a alteração de orientação das roupas. A escolha é uma adaptação inicial;
não foi realizado um experimento isolado comparando esses dois fatores.

### 3.2. Translação

`RandomTranslation(height_factor=0.1, width_factor=0.1)` desloca as imagens nos
eixos vertical e horizontal em até 10% de suas dimensões. Para 28 pixels:

```text
0,1 × 28 = 2,8 pixels
```

Isso simula pequenas mudanças de posição. Os argumentos se referem a frações da
altura e da largura, e não a um deslocamento fixo de 0,1 pixel.

### 3.3. Zoom

`RandomZoom(0.1)` introduz variações de escala, aproximando ou afastando partes
da imagem conforme o fator sorteado. Utilizamos o mesmo argumento da aula, sem
configurações adicionais para preenchimento ou interpolação.

O rótulo original é mantido nas três transformações. Por isso, os fatores devem
ser limitados: a imagem transformada ainda precisa representar a mesma classe.

Na configuração do modelo salvo, as três camadas utilizam `fill_mode='reflect'`
e `interpolation='bilinear'`, padrões mantidos na adaptação da aula. O primeiro
preenche regiões expostas nas bordas por reflexão do conteúdo da imagem, o que
pode produzir detalhes espelhados. O segundo combina pixels vizinhos ao calcular
as novas posições, podendo suavizar detalhes. Essas operações ajudam a explicar
por que uma imagem transformada pode apresentar diferenças além de sua posição.

## 4. Definição e visualização do aumento de dados

Esta é a primeira célula de código da seção da CNN:

```python
# Adaptado da Aula Prática 21, data_augmentation e sua visualização.
tf.keras.backend.clear_session()
keras.utils.set_random_seed(SEED)

data_augmentation = keras.Sequential([
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    keras.layers.RandomZoom(0.1),
], name='aumento_dados')

img = X_train[0]
fig, eixos = plt.subplots(3, 3, figsize=(8, 8))
eixos.flat[0].imshow(img.squeeze(), cmap='gray', vmin=0, vmax=1)
eixos.flat[0].set_title('Original')
eixos.flat[0].axis('off')
for i, eixo in enumerate(eixos.flat[1:], start=1):
    augmented = data_augmentation(tf.expand_dims(img, 0), training=True)
    eixo.imshow(augmented[0].numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
    eixo.set_title(f'Versão {i}')
    eixo.axis('off')
fig.suptitle(f'Aumento de dados — {NOMES_CLASSES[int(y_train[0])]}')
fig.tight_layout()
fig.savefig(RESULTADOS / 'aumento_dados_cnn.png', dpi=150, bbox_inches='tight')
plt.show()

# Em avaliação, o aumento de dados deve devolver as imagens sem transformação.
np.testing.assert_allclose(
    data_augmentation(X_val[:32], training=False).numpy(), X_val[:32],
)
print('Conferido: aumento de dados desativado em modo de avaliação.')
```

### 4.1. Preparação da sessão e aleatoriedade

`tf.keras.backend.clear_session()` limpa o estado global de construção do Keras,
como a numeração automática das camadas. `keras.utils.set_random_seed(SEED)`
controla as fontes usuais de aleatoriedade antes de criar a CNN e as transformações.
A semente ajuda no controle experimental, sem garantir reprodução exata entre
ambientes diferentes.

### 4.2. `data_augmentation`

`keras.Sequential` agrupa as três transformações na ordem definida: rotação,
translação e zoom. O nome `aumento_dados` facilita a identificação desse bloco no
resumo da rede. Ele transforma os pixels, mas não aprende pesos de classificação.

### 4.3. A imagem original e suas versões

`img = X_train[0]` seleciona uma imagem de treino, sem consultar a validação ou o
teste. `plt.subplots(3, 3, figsize=(8, 8))` cria um painel com nove posições.
A primeira mostra a imagem original, e as demais mostram oito versões sorteadas.

`tf.expand_dims(img, 0)` adiciona o eixo de lote: o formato `(28, 28, 1)` passa a
`(1, 28, 28, 1)`. A chamada com `training=True` solicita explicitamente as
transformações para essa visualização.

`augmented[0]` seleciona a única imagem do lote. `.numpy()` converte o tensor para
um array, e `.squeeze()` remove o eixo de canal de tamanho 1 para exibição em cinza.
Os limites `vmin=0` e `vmax=1` mantêm a mesma escala visual entre os painéis.

O título utiliza `NOMES_CLASSES[int(y_train[0])]` para mostrar a classe verdadeira
da imagem escolhida. `int(...)` converte o rótulo em um índice inteiro da lista.

`fig.tight_layout()` ajusta a organização, `savefig` grava a figura com resolução
de 150 dpi e `plt.show()` incorpora sua visualização às saídas do notebook.

O painel é uma inspeção de um exemplo, não uma verificação de que toda imagem
possível conservará todas as suas características após a transformação.

Na figura salva, a imagem escolhida pertence à classe `Camisa`. As versões
apresentam mudanças de posição, inclinação e escala, e algumas aproximam a peça
das bordas. Isso reforça a necessidade de inspecionar o efeito combinado das
transformações, mesmo quando seus fatores individuais são pequenos.

![Uma camisa original e oito versões produzidas pelo aumento de dados.](../resultados/aumento_dados_cnn.png)

### 4.4. Por que `training=True` não aparece em `fit`

O aumento será inserido como uma camada da própria CNN. O Keras transmite o modo
de execução às camadas: durante o treino, as transformações são ativadas; durante
a validação e a inferência, elas são desativadas.

Usamos `training=True` apenas na visualização direta, para solicitar versões
transformadas fora do treinamento. Não pré-transformamos `X_val` nem passamos
imagens aumentadas como validação.

### 4.5. Verificação com `training=False`

`np.testing.assert_allclose(...)` compara um lote de 32 imagens de validação com a
saída do aumento de dados em modo de avaliação. A verificação confirma que o
bloco devolve essas imagens sem transformação nesse modo.

Esse uso da validação confere o funcionamento da implementação. Ele não altera
pesos, não seleciona hiperparâmetros e não calcula resultados de teste.

## 5. Construção e compilação da CNN

Esta é a segunda célula da implementação:

```python
# Adaptado da Aula Prática 21, model3 (CNN com aumento de dados).
model_cnn = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    data_augmentation,
    keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu'),
    keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax'),
])

model_cnn.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model_cnn.summary()
```

### 5.1. Entrada e bloco de aumento

`keras.Input(shape=(28, 28, 1))` declara o formato de uma imagem. O tamanho do
lote não é informado nesse `shape`.

`data_augmentation` é a primeira etapa de processamento dentro da rede. A forma
da imagem permanece 28 × 28 × 1; o que muda durante o treino é sua apresentação.

### 5.2. As três camadas `Conv2D`

Uma convolução aplica um pequeno conjunto de pesos, chamado de filtro, a regiões
da imagem. O mesmo filtro é utilizado em diferentes posições, produzindo um mapa
de características. Os pesos são aprendidos durante o treino.

A primeira camada tem 8 filtros, a segunda 16 e a terceira 32. Cada filtro utiliza
uma região espacial de 3 × 3 e combina os canais da entrada que recebe.
As camadas seguintes operam sobre mapas de características, e não diretamente
sobre os pixels originais.

`activation='relu'` aplica a função `max(0, z)` ao resultado das combinações.
Essa não linearidade permite representar relações que uma sequência de operações
puramente afins não conseguiria representar da mesma maneira.

Como na aula, não acrescentamos padding para preservar as dimensões nem alteramos
o passo de deslocamento das convoluções. Neste modelo, cada convolução reduz
altura e largura em duas posições:

```text
28 × 28 × 1
  → 26 × 26 × 8
  → 24 × 24 × 16
  → 22 × 22 × 32
```

Não foram visualizados os filtros aprendidos nem atribuídos significados
específicos a cada um neste experimento. A explicação descreve o papel da operação.

### 5.3. `MaxPooling2D(pool_size=(2, 2))`

O pooling seleciona o maior valor de cada janela 2 × 2 em cada mapa de
características. Nesta arquitetura, reduz os mapas de 22 × 22 para 11 × 11,
mantendo seus 32 canais.

Essa operação reduz as dimensões espaciais e a quantidade de valores enviados
para a parte densa. Ela não possui pesos aprendidos.

### 5.4. `Flatten` e as camadas densas

`Flatten` reorganiza os mapas em um vetor com:

```text
11 × 11 × 32 = 3.872 valores
```

`Dense(64, activation='relu')` combina esses valores em 64 neurônios, como na
parte de classificação do `model3` da aula.

`Dense(10, activation='softmax')` produz uma distribuição de probabilidades para
as dez classes. A posição da maior probabilidade representa a classe prevista,
com o mesmo mapeamento de rótulos utilizado na MLP.

### 5.5. Formatos e parâmetros

Para uma convolução com viés, o número de parâmetros é:

```text
(altura do kernel × largura do kernel × canais de entrada + 1) × filtros
```

Para uma camada densa com viés, é:

```text
(quantidade de entradas + 1) × quantidade de neurônios
```

| Camada | Saída por imagem | Cálculo dos parâmetros | Total |
| --- | --- | --- | ---: |
| Aumento de dados | 28 × 28 × 1 | Sem pesos aprendidos | 0 |
| Conv2D, 8 filtros | 26 × 26 × 8 | `(3 × 3 × 1 + 1) × 8` | 80 |
| Conv2D, 16 filtros | 24 × 24 × 16 | `(3 × 3 × 8 + 1) × 16` | 1.168 |
| Conv2D, 32 filtros | 22 × 22 × 32 | `(3 × 3 × 16 + 1) × 32` | 4.640 |
| MaxPooling2D | 11 × 11 × 32 | Sem pesos aprendidos | 0 |
| Flatten | 3.872 | Sem pesos aprendidos | 0 |
| Dense, 64 neurônios | 64 | `(3.872 + 1) × 64` | 247.872 |
| Dense, 10 neurônios | 10 | `(64 + 1) × 10` | 650 |
| Total da rede | 10 | Soma das camadas | **254.410** |

A maior parte dos parâmetros está na primeira camada densa. O resumo gerado por
`model_cnn.summary()` permite conferir a arquitetura, os formatos e essas contagens.

### 5.6. Compilação

Mantemos os mesmos critérios utilizados na MLP:

- **Adam com taxa 0,001:** atualiza os pesos utilizando os gradientes da perda.
- **`sparse_categorical_crossentropy`:** compara as probabilidades com os rótulos
  inteiros de 0 a 9.
- **`accuracy`:** registra a proporção de classes previstas corretamente.

A saída já possui `softmax`. A perda recebe probabilidades, sem configuração
para receber logits. A acurácia é acompanhada, mas é a perda que orienta as
atualizações dos pesos.

## 6. Treinamento e registro do histórico

A terceira célula executa o treinamento:

```python
history_cnn = model_cnn.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=2,
)

historico_cnn = pd.DataFrame(history_cnn.history)
historico_cnn.index = pd.RangeIndex(1, len(historico_cnn) + 1, name='epoca')
historico_cnn.to_csv(RESULTADOS / 'historico_cnn.csv')
model_cnn.save(MODELOS / 'cnn.keras')

display(historico_cnn.iloc[[0, -1]])
print('Época de menor loss de validação:', historico_cnn['val_loss'].idxmin())
```

### 6.1. Argumentos de `fit`

| Argumento | Papel nesta execução |
| --- | --- |
| `X_train`, `y_train` | Imagens e rótulos utilizados para aprender os pesos. |
| `validation_data=(X_val, y_val)` | Avaliação nas mesmas imagens de validação utilizadas pela MLP. |
| `epochs=20` | Vinte passagens pelo conjunto de treino. |
| `batch_size=32` | Até 32 imagens por lote de atualização. |
| `verbose=2` | Resumo do progresso e das métricas por época. |

Com 48.000 imagens e lotes de 32, cada época tem 1.500 lotes de treino. Ao longo
de 20 épocas, são 30.000 atualizações. A geração das imagens transformadas faz
parte do processamento de treino; a validação não executa as transformações.

A execução termina ao completar as 20 épocas. Não há parada antecipada ou
restauração automática dos pesos de uma época anterior.

### 6.2. Histórico e DataFrame

`history_cnn` é o objeto `History` retornado por `fit`. Seu atributo `.history`
contém quatro listas, cada uma com um valor por época:

- `loss`: perda de treino;
- `accuracy`: acurácia de treino;
- `val_loss`: perda de validação;
- `val_accuracy`: acurácia de validação.

`pd.DataFrame(history_cnn.history)` organiza essas listas em uma tabela chamada
`historico_cnn`. Cada linha corresponde a uma época.

`pd.RangeIndex(1, len(historico_cnn) + 1, name='epoca')` define os números de época
de 1 a 20 como índice da tabela. O limite final não é incluído pelo `RangeIndex`,
por isso o código acrescenta 1 ao número de linhas.

`to_csv(...)` salva as métricas em `resultados/historico_cnn.csv`, com mais casas
decimais do que o resumo mostrado durante o treinamento.

### 6.3. Modelo e resumo dos resultados

`model_cnn.save(MODELOS / 'cnn.keras')` salva o modelo depois do treinamento,
incluindo a arquitetura, o bloco de aumento de dados e os pesos aprendidos.
O arquivo corresponde à última época, não automaticamente à melhor época de
validação.

`display(historico_cnn.iloc[[0, -1]])` exibe a primeira e a última linha da tabela:
`0` e `-1` são posições, que correspondem às épocas 1 e 20.

`historico_cnn['val_loss'].idxmin()` identifica a época de menor perda de validação.
Essa consulta não modifica nem restaura pesos. Se a menor perda ocorrer antes da
época 20, os pesos desse momento não passam a ser os do arquivo salvo.

Os caminhos `RESULTADOS` e `MODELOS` são objetos `Path` definidos na preparação.
O operador `/`, nessas expressões, junta a pasta ao nome do arquivo.

## 7. Como interpretar treino e validação com aumento de dados

As métricas dos dois conjuntos são calculadas em condições diferentes:

| Aspecto | Treino da CNN | Validação da CNN |
| --- | --- | --- |
| Imagens | Versões transformadas aleatoriamente | Imagens normalizadas sem aumento |
| Atualização dos pesos | Ocorre a cada lote | Não ocorre |
| Momento das métricas | Acumuladas durante as atualizações | Calculadas ao final da época |

Por isso, não é necessário que a acurácia de treino seja maior que a de validação.
O aumento pode tornar os exemplos apresentados no treino mais difíceis, enquanto
a validação utiliza as imagens sem transformações. Além disso, os pesos mudam
ao longo da acumulação das métricas de treino.

A diferença entre as duas curvas precisa ser interpretada junto com essas
condições. Não basta comparar uma única acurácia para concluir se houve vazamento,
sobreajuste ou subajuste.

As definições das métricas permanecem as mesmas do item 1:

$$
\mathrm{acurácia} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}(\hat{y}_i=y_i),
\qquad
\mathrm{loss} = -\frac{1}{N}\sum_{i=1}^{N}\ln p_{i,y_i}.
$$

A acurácia conta classes previstas corretamente. A perda considera a
probabilidade atribuída à classe verdadeira. As probabilidades podem mudar sem
que a classe mais provável mude; por isso, as métricas não precisam ser
proporcionais.

## 8. Gráficos e comparação com a MLP

A quarta célula produz as curvas e a tabela de comparação:

```python
# Adaptado dos gráficos de history3 da Aula Prática 21.
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].plot(historico_cnn.index, historico_cnn['loss'], label='Treino')
ax[0].plot(historico_cnn.index, historico_cnn['val_loss'], label='Validação')
ax[0].set_title('CNN com aumento de dados — perda por época')
ax[0].set_xlabel('Época')
ax[0].set_ylabel('Loss (entropia cruzada)')
ax[0].grid(True)
ax[0].legend()

ax[1].plot(historico_cnn.index, historico_cnn['accuracy'], label='Treino')
ax[1].plot(historico_cnn.index, historico_cnn['val_accuracy'], label='Validação')
ax[1].set_title('CNN com aumento de dados — acurácia por época')
ax[1].set_xlabel('Época')
ax[1].set_ylabel('Acurácia')
ax[1].set_ylim(0, 1.05)
ax[1].grid(True)
ax[1].legend()

fig.tight_layout()
fig.savefig(RESULTADOS / 'curvas_cnn.png', dpi=150, bbox_inches='tight')
plt.show()

comparacao_modelos = pd.DataFrame([
    historico_mlp.iloc[-1], historico_cnn.iloc[-1],
], index=['MLP', 'CNN com aumento de dados'])
comparacao_modelos.index.name = 'modelo'
comparacao_modelos['epocas'] = [len(historico_mlp), len(historico_cnn)]
display(comparacao_modelos)
comparacao_modelos.to_csv(RESULTADOS / 'comparacao_mlp_cnn.csv')
```

### 8.1. Elementos dos gráficos

`plt.subplots(1, 2, figsize=(14, 5))` cria uma figura com dois gráficos lado a
lado. `fig` é a figura completa; `ax[0]` representa o gráfico de perda, e `ax[1]`
o de acurácia.

Cada chamada de `plot` utiliza `historico_cnn.index` no eixo horizontal, isto é,
as épocas de 1 a 20. O segundo argumento fornece a métrica:

| Gráfico | Primeira curva | Segunda curva |
| --- | --- | --- |
| Perda | `historico_cnn['loss']` | `historico_cnn['val_loss']` |
| Acurácia | `historico_cnn['accuracy']` | `historico_cnn['val_accuracy']` |

`label` nomeia as curvas, e `legend()` exibe a legenda. `set_title`, `set_xlabel`
e `set_ylabel` identificam o conteúdo e os eixos. `grid(True)` facilita a leitura
dos valores.

`set_ylim(0, 1.05)` fixa a escala da acurácia e deixa uma margem acima de 1. Essa
margem é apenas visual; a acurácia não pode superar 100%.

`fig.tight_layout()` ajusta os espaços da figura. `fig.savefig(...)` exporta os
gráficos para `resultados/curvas_cnn.png`, com resolução de 150 dpi e ajuste da
área ao conteúdo. `plt.show()` exibe a figura no notebook, onde a saída é salva.

### 8.2. Construção da tabela comparativa

`historico_mlp.iloc[-1]` e `historico_cnn.iloc[-1]` selecionam as métricas da
última época de cada modelo. `pd.DataFrame(...)` reúne essas duas linhas em uma
nova tabela, com os nomes dos modelos como índice.

`comparacao_modelos.index.name = 'modelo'` dá um nome à coluna de identificação
ao exportar a tabela. A coluna `epocas` registra o número de linhas de cada
histórico, ou seja, a quantidade de épocas executadas.

A tabela é exibida no notebook e salva em
`resultados/comparacao_mlp_cnn.csv`. Ela utiliza os números medidos nos históricos,
sem preencher resultados manualmente.

### 8.3. O que essa comparação permite concluir

Os resultados de validação são comparáveis porque utilizam os mesmos exemplos,
rótulos, normalização e métricas. Os dois modelos também utilizam 20 épocas,
lotes de 32 e Adam com taxa 0,001.

Entretanto, as configurações diferem em vários aspectos:

- A MLP utiliza diretamente os pixels achatados, uma camada oculta com `tanh` e
  101.770 parâmetros.
- A CNN utiliza convoluções, pooling, uma camada densa com ReLU, 254.410
  parâmetros e aumento de dados durante o treino.

Uma diferença de desempenho descreve o resultado dessas configurações completas.
Ela não isola o efeito das convoluções, da ativação, da quantidade de parâmetros
ou do aumento de dados.

Em particular, não foi treinada esta mesma CNN sem aumento de dados nesta etapa.
Portanto, a comparação com a MLP não demonstra, por si só, quanto do resultado foi
causado pelo aumento de dados.

Ter a mesma quantidade de épocas também não implica o mesmo custo de execução.
Além de gerar as transformações, a CNN aplica filtros em muitas posições das
imagens. A quantidade de operações não depende apenas da quantidade de pesos,
pois os pesos de uma convolução são reutilizados em diferentes posições. Os logs
do notebook permitem observar os tempos por época desta execução, sem constituir
um benchmark entre dispositivos ou ambientes.

## 9. Critérios para analisar as curvas

O item 2 deve considerar tanto as métricas quanto sua evolução:

| Comportamento | Interpretação possível |
| --- | --- |
| Perda diminui e acurácia melhora na validação | Evidência de aprendizado que também aparece em imagens fora das atualizações. |
| Treino melhora enquanto a perda de validação aumenta persistentemente | Indício de sobreajuste. |
| Resultados fracos nos dois conjuntos e pouca melhora | Pode haver subajuste ou dificuldade de otimização. |
| Validação tem acurácia maior que o treino aumentado | Pode ser compatível com o aumento tornando o treino mais difícil. |
| Acurácia quase estável e perda variável | As probabilidades podem estar mudando sem alterar muitas classes previstas. |

Oscilações isoladas não bastam para um diagnóstico. Também não é possível
garantir generalização a qualquer situação real a partir de uma única divisão e
execução do Fashion-MNIST.

## 10. Resultados medidos e análise

**Finalidade do aumento de dados.** As rotações, translações e variações de escala
apresentam novas versões das imagens durante o treino, buscando reduzir a
dependência de posições e orientações específicas. O painel foi inspecionado:
a camisa continua reconhecível, mas algumas versões se aproximam das bordas.
Transformações pequenas também podem alterar detalhes; essa técnica precisa ser
avaliada junto com o desempenho, sem garantir melhoria por si só. A verificação
em modo de avaliação confirmou a ausência de transformação no lote conferido.

**Resultados medidos.** A CNN foi treinada por 20 épocas. A tabela mostra a
primeira e a última; nesta execução, a época 20 também apresentou a menor perda
e a maior acurácia de validação.

| Momento | Loss de treino | Acurácia de treino | Loss de validação | Acurácia de validação |
| --- | --- | --- | --- | --- |
| Época 1 | 0,6964 | 73,59% | 0,5978 | 78,01% |
| Época 20 | 0,3022 | 88,79% | 0,3254 | 88,63% |

**O desempenho é satisfatório?** Como configuração inicial, sim: a validação
atingiu 88,63% de acertos, bem acima dos 10% de um classificador que sempre
escolhesse uma única classe nessa divisão balanceada. Essa referência de 10%
decorre da distribuição dos rótulos, não do treinamento de outro modelo.
O enunciado não exige uma acurácia mínima, e o resultado não demonstra que a
configuração seja ótima.

**Sobreajuste ou subajuste.** A loss de treino caiu de 0,6964 para 0,3022, e a
de validação de 0,5978 para 0,3254. Houve oscilações: na época 17, a loss de
validação subiu para 0,4039, mas depois voltou a cair, atingindo o mínimo na
época 20. Não observamos a tendência persistente de piora da loss de validação
que apareceu na MLP após a época 6. Isso não prova ausência de sobreajuste.
Também não há um padrão claro de subajuste severo: o modelo melhora no treino
e na validação ao longo da execução.

A diferença final entre as acurácias é de aproximadamente 0,15 ponto percentual.
Esse valor precisa ser interpretado com cuidado: o treino da CNN recebe imagens
transformadas e acumula métricas durante as atualizações; a validação recebe
imagens sem aumento e é medida ao final da época. A diferença menor que a da
MLP não isola nem comprova o efeito do aumento de dados.

**Perda e acurácia.** A acurácia considera a classe de maior probabilidade,
enquanto a perda considera a probabilidade da classe verdadeira, conforme as
fórmulas do item 1. Na validação da CNN, entre as épocas 9 e 17, a acurácia
passou de 86,04% para 86,46%, enquanto a perda aumentou de 0,3966 para 0,4039.
Esse exemplo real mostra que uma pequena melhora na proporção de acertos pode
coexistir com piora na perda, sem proporcionalidade entre as métricas.

**Comparação com a MLP na última época:**

| Modelo | Loss de treino | Acurácia de treino | Loss de validação | Acurácia de validação |
| --- | --- | --- | --- | --- |
| MLP | 0,1704 | 93,83% | 0,3603 | 88,60% |
| CNN com aumento de dados | 0,3022 | 88,79% | 0,3254 | 88,63% |

A acurácia de validação da CNN foi apenas cerca de **0,03 ponto percentual**
maior, equivalente a **quatro acertos adicionais nas 12.000 imagens**. Essa
diferença pequena, obtida em uma única execução, não sustenta uma conclusão de
superioridade consistente. A loss final de validação foi menor na CNN, indicando
melhor resultado nesse critério para as configurações e épocas comparadas.

A MLP teve acurácia de treino maior, mas o treino da CNN inclui transformações
que podem dificultar os exemplos. Arquitetura, ativação, quantidade de parâmetros
e aumento de dados mudam juntos. Não treinamos esta mesma CNN sem aumento;
portanto, não podemos atribuir isoladamente a ele as diferenças observadas.

**Custo do treinamento.** Nos logs desta execução em CPU, as épocas da CNN
levaram aproximadamente 24 a 28 segundos, enquanto as da MLP levaram 3 a 4
segundos. São tempos arredondados do Keras, sujeitos à carga e ao ambiente, e
não um benchmark. A CNN também tem mais parâmetros: 254.410, contra 101.770
da MLP, além das operações espaciais e das transformações.

**Generalização e limites.** A validação indica aprendizado útil para imagens
que não atualizaram os pesos, com desempenho de acurácia próximo ao da MLP.
O resultado se limita a esta divisão e execução. A semente não garante reprodução
exata entre ambientes. O arquivo `modelos/cnn.keras` contém os pesos da época 20,
que neste histórico coincidiu com os melhores resultados de validação.
O conjunto de teste permanece reservado; a otimização e a avaliação final ainda
serão realizadas nos itens 3 e 4.

![Curvas reais de perda e acurácia da CNN durante as 20 épocas.](../resultados/curvas_cnn.png)

## 11. Arquivos gerados e leitura no notebook

| Arquivo | Conteúdo |
| --- | --- |
| `Trabalho_02_Redes_Neurais.ipynb` | Implementação, explicações, logs, figuras, tabela e análise com saídas salvas. |
| `resultados/aumento_dados_cnn.png` | Imagem original de treino e oito versões transformadas. |
| `resultados/historico_cnn.csv` | Perda e acurácia de treino e validação por época. |
| `resultados/curvas_cnn.png` | Dois gráficos com as quatro curvas. |
| `resultados/comparacao_mlp_cnn.csv` | Métricas da última época da MLP e da CNN. |
| `modelos/cnn.keras` | Modelo com os pesos da última época. |

As pastas de dados, resultados e modelos possuem arquivos ignorados pelo Git.
Uma cópia obtida somente pelo versionamento pode não conter os arquivos externos.
As figuras, tabelas e logs também são preservados nas saídas do notebook para a
entrega.

Para acompanhar a implementação, leia a seção 5 na ordem dos passos: aumento de
dados, construção, treinamento, gráficos e análise. As células dependem da
preparação anterior, e a tabela comparativa depende do histórico da MLP.

Reiniciar o kernel apaga as variáveis da memória. Os arquivos permanecem no disco,
mas isso não recria automaticamente os objetos Python utilizados pelas células.
Para executar tudo novamente, siga a ordem do notebook desde a preparação dos
dados. Para estudar os resultados existentes, basta consultar as saídas salvas.

Uma nova execução pode substituir os arquivos gerados e produzir resultados
diferentes. Nesse caso, as análises devem ser conferidas em relação ao novo
histórico antes de apresentar o trabalho.

## 12. Referências e registro de apoio

- **Profa. Nathalia da Cruz Alves:**
  [enunciado do trabalho](<Trabalho Prático 2 - Redes Neurais.pdf>) e
  [Aula 21 — CNN](../Aulas_Praticas/Aula_21_CNN_MNIST_ESTUDANTE.ipynb), em especial
  a seção de aumento de dados e o `model3`. A autoria do material foi preservada.
- **Notebook da equipe:**
  [Trabalho_02_Redes_Neurais.ipynb](../Trabalho_02_Redes_Neurais.ipynb), seção 5,
  fonte do código e das saídas explicadas aqui.
- **Dados:** origem, licença e referência do Fashion-MNIST no
  [documento local dos dados](../dados/README.md).
- **OpenAI Codex:** apoio à adaptação da CNN e do aumento de dados da aula,
  execução, visualizações, comparação das métricas e redação desta documentação.
  A equipe deve revisar o conteúdo e compreender as decisões e os resultados
  para apresentar o trabalho.
