# Item 1 — Implementação, treinamento e análise da MLP

Este documento explica a implementação do item 1 do Trabalho 02 de Sistemas
Inteligentes: classificação de imagens do Fashion-MNIST com uma rede MLP, sem
camadas convolucionais. Ele acompanha o código da seção 4 do
[notebook da equipe](../Trabalho_02_Redes_Neurais.ipynb) e as saídas reais salvas
nessa seção.

O objetivo é permitir que a equipe entenda o que cada elemento faz, por que foi
utilizado e como interpretar os resultados. As três células de código da MLP são
reproduzidas abaixo e explicadas em partes. A execução deve continuar sendo feita
no notebook, que reúne os dados, o código e os resultados exigidos para a entrega.

Para localizar uma parte específica:

- [1. Requisitos do item 1](#1-o-que-o-item-1-exige)
- [2. Material da aula e adaptações](#2-material-da-aula-utilizado-como-base)
- [3. Visão geral do processamento](#3-visão-geral-do-processamento)
- [4. Dados e variáveis](#4-dados-e-variáveis-de-que-a-mlp-depende)
- [5. Construção da rede](#5-construção-e-compilação-código-completo)
- [6. Compilação, otimizador e métricas](#6-compilação-otimizador-e-métricas)
- [7. Treinamento](#7-treinamento-e-registro-código-completo)
- [8. Histórico e arquivos salvos](#8-histórico-tabela-e-modelo-salvo)
- [9. Código dos gráficos](#9-gráficos-código-completo)
- [10. Explicação dos gráficos](#10-explicação-de-cada-elemento-dos-gráficos)
- [11. Leitura das curvas](#11-como-ler-as-curvas)
- [12. Resultados e análise](#12-resultados-reais-e-análise)
- [13. Configuração e arquivos](#13-configuração-e-arquivos-do-experimento)
- [14. Como acompanhar no notebook](#14-como-acompanhar-a-implementação-no-notebook)
- [15. Referências e uso de IA](#15-referências-e-registro-de-apoio)

## 1. O que o item 1 exige

Conforme o [enunciado](<Trabalho Prático 2 - Redes Neurais.pdf>), esta etapa deve:

1. Implementar e executar uma MLP sem convoluções.
2. Acompanhar a perda, também chamada de `loss`, e a acurácia de treino e validação.
3. Explicar como essas métricas são calculadas e por que não são proporcionais.
4. Apresentar gráficos das duas métricas por época.
5. Discutir se o desempenho é satisfatório e o que a validação sugere sobre
   generalização.

Nesta implementação, o treinamento foi executado por 20 épocas. O resultado final
foi de **93,83% de acurácia no treino e 88,60% na validação**. Esses valores serão
analisados na seção 12, junto com as perdas e o comportamento das curvas.

## 2. Material da aula utilizado como base

A referência prática é a
[Aula 19 — Perceptron](../Aulas_Praticas/Aula_19_Perceptron_Estudante.ipynb),
de autoria da **Profa. Nathalia da Cruz Alves**, especialmente:

- A seção **Multilayer Perceptron (MLP)**, que explica camadas ocultas e não
  linearidade.
- A função **`treinar_mlp`**, que constrói a rede com `tf.keras.Sequential`, usa
  camadas `Dense`, compila com Adam e treina com `fit`.
- A função **`plot_metrics`**, que apresenta perda e acurácia em dois gráficos
  lado a lado.

As orientações do repositório mencionam um arquivo com o nome
`Aula_20_Perceptron_Estudante.ipynb`. Na pasta fornecida, a implementação de MLP
está no arquivo da Aula 19 indicado acima.

O exemplo da aula resolve o XOR, que possui duas entradas e uma saída binária.
O trabalho classifica roupas em dez classes; por isso, algumas adaptações são
necessárias:

| Elemento | MLP do XOR na aula | MLP deste trabalho | Motivo da adaptação |
| --- | --- | --- | --- |
| Entrada | Duas variáveis | 784 pixels por imagem | Cada imagem tem 28 × 28 pixels. |
| Organização da entrada | Vetor com duas posições | `Input` e `Flatten` | As imagens precisam ser reorganizadas em vetores. |
| Camada oculta | Quantidade configurável; exemplo com 2 neurônios | 128 neurônios | Configuração inicial para o problema de imagens. |
| Ativação oculta | `tanh` | `tanh` | Mantém a ativação da MLP da aula. |
| Saída | 1 neurônio com `sigmoid` | 10 neurônios com `softmax` | É necessário distinguir dez classes mutuamente exclusivas. |
| Perda | `binary_crossentropy` | `sparse_categorical_crossentropy` | Os rótulos agora são inteiros de 0 a 9. |
| Otimizador | Adam | Adam | Mantém o otimizador da MLP da aula. |
| Taxa de aprendizagem | 0,1 como padrão da função | 0,001 | Escolha inicial menor para o problema de imagens. |
| Épocas | 1.000 como padrão da função | 20 | Orçamento inicial de treinamento do experimento. |
| Validação | Não informada na função do XOR | `validation_data=(X_val, y_val)` | O enunciado exige comparar treino e validação. |
| Gráficos | Métricas de treino | Métricas de treino e validação | É necessário acompanhar a generalização durante o treinamento. |

Os valores 128, 0,001, 20 e 32, usado como tamanho do lote, são escolhas desta
implementação. Eles **não foram determinados por uma busca de hiperparâmetros**,
e a execução não demonstra que sejam as melhores escolhas possíveis.

No trabalho, a construção, o treinamento e os gráficos foram separados em células
curtas para facilitar a leitura. A sequência de operações continua sendo a da
aula: construir, compilar, treinar e observar as métricas. A autoria do material
de referência permanece atribuída à professora.

## 3. Visão geral do processamento

Uma imagem percorre o seguinte caminho dentro da rede:

```text
Imagem normalizada: 28 × 28 × 1
            ↓
Flatten: vetor com 784 pixels
            ↓
Dense: 128 neurônios com tanh
            ↓
Dense: 10 neurônios com softmax
            ↓
Probabilidades das 10 classes
            ↓
Classe prevista: posição da maior probabilidade
```

Durante o treino, a classe verdadeira é utilizada para calcular a perda. O Keras
calcula os gradientes por retropropagação, e o Adam utiliza esses gradientes para
atualizar os pesos. Durante a validação, as previsões e métricas são calculadas
sem atualizar os pesos.

## 4. Dados e variáveis de que a MLP depende

### 4.1. O problema de classificação

Cada imagem representa uma peça de vestuário em escala de cinza. A rede deve
escolher uma das dez classes, preservando a numeração do Fashion-MNIST:

| Rótulo | Classe utilizada no notebook |
| --- | --- |
| 0 | Camiseta/top |
| 1 | Calça |
| 2 | Suéter |
| 3 | Vestido |
| 4 | Casaco |
| 5 | Sandália |
| 6 | Camisa |
| 7 | Tênis |
| 8 | Bolsa |
| 9 | Bota de cano curto |

Os números identificam categorias. O rótulo 9 não representa uma roupa maior,
melhor ou mais próxima da classe 8 do que da classe 0.

### 4.2. Treino, validação e teste

O Fashion-MNIST fornecido tem 60.000 imagens no treino oficial e 10.000 no teste.
O notebook separa 20% do treino oficial para validação:

| Conjunto | Quantidade | Papel nesta etapa |
| --- | ---: | --- |
| Treino | 48.000 | Calcular a perda e atualizar os pesos. |
| Validação | 12.000 | Acompanhar o desempenho em imagens que não atualizam os pesos. |
| Teste | 10.000 | Permanece reservado para a avaliação final do trabalho. |

A separação utiliza `train_test_split`, com `test_size=0.2`, `random_state=42`
e `stratify=y_oficial`. A estratificação preserva a proporção das classes.
Nesta divisão, cada classe tem 4.800 imagens de treino e 1.200 de validação.

Os índices são salvos em `resultados/divisao_dados.npz`. Os experimentos seguintes
devem utilizar a mesma divisão para permitir uma comparação coerente.

### 4.3. Normalização e dimensões

A preparação existente no notebook é:

```python
X_train = X_oficial[indices_treino].astype('float32')[..., np.newaxis] / 255.0
X_val = X_oficial[indices_validacao].astype('float32')[..., np.newaxis] / 255.0
y_train = y_oficial[indices_treino]
y_val = y_oficial[indices_validacao]
```

Cada parte tem uma função:

- `X_oficial[indices_treino]` seleciona somente as imagens pertencentes ao treino.
  Para validação, a seleção utiliza `indices_validacao`.
- `.astype('float32')` converte os pixels inteiros para números de ponto
  flutuante, utilizados nas operações da rede.
- `[..., np.newaxis]` acrescenta uma dimensão no final. Uma imagem passa de
  `(28, 28)` para `(28, 28, 1)`. O número 1 representa o único canal de cinza.
  O mesmo formato poderá ser aproveitado pela CNN.
- `/ 255.0` transforma a escala dos pixels de `[0, 255]` para `[0, 1]`. Assim,
  a rede recebe valores em uma escala menor. Essa transformação fixa não calcula
  estatísticas a partir da validação.
- `y_oficial[...]` seleciona os rótulos correspondentes às mesmas imagens.
  Os rótulos permanecem inteiros de 0 a 9 e não são divididos por 255.

As dimensões resultantes são:

| Variável | Formato | Conteúdo |
| --- | --- | --- |
| `X_train` | `(48000, 28, 28, 1)` | Imagens de treino normalizadas. |
| `y_train` | `(48000,)` | Um rótulo inteiro por imagem de treino. |
| `X_val` | `(12000, 28, 28, 1)` | Imagens de validação normalizadas. |
| `y_val` | `(12000,)` | Um rótulo inteiro por imagem de validação. |

A MLP não utiliza aumento de dados nesta etapa. Normalizar os pixels não equivale
a gerar imagens transformadas: apenas altera a escala numérica da entrada.

### 4.4. Bibliotecas e variáveis auxiliares

As importações e as pastas são definidas antes da seção 4 do notebook:

| Elemento | Uso na implementação da MLP |
| --- | --- |
| `tensorflow as tf` | Construir as camadas, compilar a rede e executar o treinamento. |
| `keras`, importado de `tensorflow` | Acessar as utilidades do Keras, como o controle de sementes. |
| `numpy as np` | Preparar os arrays de imagens e rótulos. |
| `pandas as pd` | Organizar o histórico em uma tabela e salvá-lo em CSV. |
| `matplotlib.pyplot as plt` | Criar e salvar os gráficos de perda e acurácia. |
| `display` | Exibir a tabela de resultados no notebook. |
| `SEED` | Semente definida como 42. |
| `RESULTADOS` | Caminho para a pasta `resultados/`. |
| `MODELOS` | Caminho para a pasta `modelos/`. |

`RESULTADOS` e `MODELOS` são objetos `Path`, da biblioteca padrão `pathlib`.
Por isso, expressões como `RESULTADOS / 'historico_mlp.csv'` representam a junção
da pasta com o nome do arquivo. Nesse contexto, `/` é uma operação sobre caminhos,
não uma divisão numérica.

## 5. Construção e compilação: código completo

Esta é a primeira célula de código da seção 4:

```python
# Adaptado da Aula Prática 19, treinar_mlp.
tf.keras.backend.clear_session()
keras.utils.set_random_seed(SEED)

model_mlp = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model_mlp.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

model_mlp.summary()
```

### 5.1. Limpeza da sessão e semente

`tf.keras.backend.clear_session()` limpa o estado global mantido pelo Keras para
a construção dos modelos, como a numeração automática das camadas. Essa chamada
também aparece na função `treinar_mlp` da aula. Ela não recarrega os dados e não
substitui o treinamento.

`keras.utils.set_random_seed(SEED)` configura as sementes das fontes usuais de
aleatoriedade utilizadas pelo Keras, incluindo Python, NumPy e TensorFlow neste
ambiente. A chamada é feita antes de construir a MLP, pois os pesos iniciais
dependem de inicialização aleatória.

O uso da semente ajuda a controlar a comparação entre execuções. Ele não garante
resultados exatamente iguais entre diferentes versões, dispositivos ou operações
com comportamento não determinístico.

### 5.2. `tf.keras.Sequential`

`Sequential` organiza as camadas em sequência: a saída de uma camada se torna a
entrada da seguinte. A estrutura é suficiente para esta MLP, que possui um único
caminho entre entrada e saída.

`model_mlp` é a variável que guarda o modelo. Ela será usada para compilar,
treinar, mostrar a arquitetura e salvar os pesos aprendidos.

### 5.3. `Input(shape=(28, 28, 1))`

`Input` declara o formato de uma imagem recebida pela rede:

- 28 linhas;
- 28 colunas;
- 1 canal de cinza.

A quantidade de imagens do lote não faz parte desse `shape`. A rede pode receber
lotes com diferentes quantidades de imagens. No resumo, essa dimensão aparece
como `None`, por exemplo em `(None, 784)`.

`Input` não aprende pesos. Sua função é declarar a entrada esperada.

### 5.4. `Flatten()`

A camada `Flatten` reorganiza cada imagem de `(28, 28, 1)` para `(784,)`, porque:

```text
28 × 28 × 1 = 784
```

Ela não aprende parâmetros, não calcula filtros e não altera os valores dos
pixels. Apenas muda a organização das dimensões para que a próxima camada densa
receba um vetor.

O vetor continua contendo os pixels em posições fixas, mas a MLP não utiliza
filtros locais nem compartilhamento de pesos como uma CNN. Cada neurônio da
primeira camada densa recebe todos os 784 valores da imagem.

### 5.5. `Dense(128, activation='tanh')`

`Dense` é uma camada totalmente conectada. Cada um dos seus 128 neurônios combina
todos os valores de entrada com pesos e acrescenta um viés, também chamado de
`bias`.

Para o neurônio `j`, essa operação pode ser escrita como:

$$
z_j = \sum_{k=1}^{784} w_{kj}x_k + b_j,
\qquad h_j = \tanh(z_j).
$$

Nessa expressão:

- `x_k` é o valor de um pixel normalizado;
- `w_kj` é o peso que liga esse pixel ao neurônio;
- `b_j` é o viés do neurônio;
- `z_j` é a combinação antes da ativação;
- `h_j` é a saída do neurônio depois de aplicar `tanh`.

Os pesos e vieses são parâmetros aprendidos durante o treino. A quantidade de
neurônios, 128, é um hiperparâmetro escolhido antes do treino.

A função `tanh` produz valores entre -1 e 1 e introduz não linearidade. Sem uma
ativação não linear entre as camadas densas, a composição dessas transformações
afins continuaria equivalente a uma única transformação afim, limitando as
relações que a rede pode representar.

A ativação foi mantida para seguir a MLP da aula. Isso não constitui evidência de
que `tanh` seja superior a outras ativações no Fashion-MNIST. Essa comparação não
foi executada nesta etapa.

### 5.6. `Dense(10, activation='softmax')`

A camada de saída recebe os 128 valores produzidos pela camada oculta e calcula
dez valores, um para cada classe. A função `softmax` transforma os valores antes
da ativação, chamados de logits, em uma distribuição de probabilidades:

$$
p_c = \frac{e^{z_c}}{\sum_{j=0}^{9}e^{z_j}}.
$$

Cada `p_c` corresponde à classe de rótulo `c`, e as dez probabilidades somam 1,
considerando a precisão numérica dos cálculos.

A classe prevista é o índice da maior probabilidade:

$$
\hat{y} = \operatorname{argmax}_{c \in \{0,\ldots,9\}} p_c.
$$

Por exemplo, se a posição 7 tiver a maior probabilidade, a classe prevista será
`Tênis`. Esse é um exemplo de interpretação da saída, não uma previsão medida
para uma imagem específica do experimento.

Usamos `softmax` porque cada imagem tem uma única classe entre dez possibilidades.
A saída com um neurônio e `sigmoid` do XOR representa uma decisão binária e não
atende diretamente a esse formato de classificação.

### 5.7. Quantidade de parâmetros e `summary()`

Em uma camada densa com `n` entradas e `m` neurônios, há `n × m` pesos e `m`
vieses. Portanto:

| Parte da rede | Cálculo | Parâmetros aprendidos |
| --- | --- | ---: |
| `Input` | Apenas declara a entrada | 0 |
| `Flatten` | Apenas reorganiza dimensões | 0 |
| Camada oculta | `784 × 128 + 128` | 100.480 |
| Camada de saída | `128 × 10 + 10` | 1.290 |
| Total | `100.480 + 1.290` | **101.770** |

`model_mlp.summary()` exibe a arquitetura, os formatos de saída e as quantidades
de parâmetros. O resumo salvo no notebook confirma 101.770 parâmetros treináveis
e nenhum parâmetro não treinável da rede, nesse momento anterior ao treinamento.

## 6. Compilação, otimizador e métricas

### 6.1. O papel de `compile`

`model_mlp.compile(...)` configura como a rede será treinada e acompanhada:

- `optimizer` define como os parâmetros serão atualizados;
- `loss` define o objetivo que o treinamento procura reduzir;
- `metrics` define as medidas adicionais registradas.

Compilar não significa treinar. Os dados só são utilizados no treinamento quando
a célula seguinte chama `fit`.

### 6.2. Adam e `learning_rate=0.001`

O Adam utiliza os gradientes da perda e mantém estimativas de seus momentos para
ajustar as atualizações dos parâmetros. Em termos práticos, os gradientes indicam
como os pesos influenciam a perda, e o otimizador utiliza essa informação para
modificá-los.

`learning_rate` é a taxa de aprendizagem, que controla a escala dessas
atualizações. A taxa 0,001 foi configurada explicitamente. Uma taxa muito grande
pode dificultar a estabilização do treino; uma taxa muito pequena pode exigir
mais atualizações para melhorar o modelo.

O uso de Adam segue a aula. A redução da taxa em relação ao exemplo XOR é uma
escolha inicial da adaptação; não houve um experimento que isolasse seu efeito.

### 6.3. `sparse_categorical_crossentropy`

Essa perda é a entropia cruzada para classificação em várias classes com rótulos
inteiros. O termo `sparse` se refere à representação dos rótulos, e não à presença
de pixels vazios nas imagens.

Neste trabalho, uma imagem da classe `Tênis` tem rótulo `7`. Não é necessário
transformá-lo em um vetor one-hot com dez posições para utilizar essa perda.

Para uma imagem cuja classe verdadeira é `y_i`, a perda é:

$$
\ell_i = -\ln p_{i,y_i}.
$$

Para `N` imagens, a perda média é:

$$
L = -\frac{1}{N}\sum_{i=1}^{N}\ln p_{i,y_i}.
$$

`p_{i,y_i}` é a probabilidade atribuída à classe verdadeira da imagem `i`.
Quanto menor essa probabilidade, maior a penalização. No treinamento, o Keras
calcula a perda dos lotes e utiliza suas derivadas para ajustar os parâmetros.

A saída já contém `softmax`, portanto a perda recebe probabilidades. Não foi
configurado `from_logits=True`, opção que corresponderia a uma saída sem essa
conversão para probabilidades.

A perda não é uma porcentagem de erro. Por exemplo, `loss = 0,3603` não significa
36,03% de imagens erradas.

### 6.4. `metrics=['accuracy']`

A acurácia compara a classe de maior probabilidade com o rótulo verdadeiro:

$$
\mathrm{acurácia} = \frac{1}{N}
\sum_{i=1}^{N}\mathbf{1}(\hat{y}_i = y_i).
$$

A função indicadora vale 1 quando a previsão está correta e 0 quando está errada.
Uma acurácia de 0,886 equivale a 88,60% de acertos.

Neste formato de entrada e saída, o acompanhamento de `accuracy` usa os rótulos
inteiros e a classe mais provável entre as dez saídas. A acurácia é registrada
para interpretação; o objetivo usado pelo otimizador é a perda.

### 6.5. Por que perda e acurácia não são proporcionais

A acurácia considera a identidade da classe mais provável. A perda considera a
probabilidade atribuída à classe verdadeira. A confiança pode mudar mesmo quando
a classe prevista permanece igual.

O exemplo abaixo é apenas didático; seus números não são saídas do treinamento:

| Situação | Probabilidade da classe verdadeira | Ela é a classe mais provável? | Acerto da imagem | Perda aproximada |
| --- | ---: | --- | ---: | ---: |
| A | 0,4 | Sim | 1 | `-ln(0,4) = 0,916` |
| B | 0,8 | Sim | 1 | `-ln(0,8) = 0,223` |

Nas duas situações, a imagem é classificada corretamente. A acurácia dessa imagem
permanece igual, mas a perda diminui. Por outro lado, uma imagem errada à qual a
rede atribua probabilidade muito baixa para a classe verdadeira pode contribuir
com uma perda alta.

Como as métricas são agregadas sobre muitas imagens, a acurácia pode melhorar
ligeiramente enquanto a perda aumenta. Essa situação ocorreu na validação desta
MLP e será discutida com os valores reais na seção 12.

## 7. Treinamento e registro: código completo

A segunda célula da MLP executa o treinamento e registra seus resultados:

```python
history_mlp = model_mlp.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32,
    verbose=2,
)

historico_mlp = pd.DataFrame(history_mlp.history)
historico_mlp.index = pd.RangeIndex(1, len(historico_mlp) + 1, name='epoca')
historico_mlp.to_csv(RESULTADOS / 'historico_mlp.csv')
model_mlp.save(MODELOS / 'mlp.keras')

display(historico_mlp.iloc[[0, -1]])
print('Época de menor loss de validação:', historico_mlp['val_loss'].idxmin())
```

### 7.1. `model_mlp.fit`

`fit` executa o treinamento da rede já construída e compilada. Os argumentos
`X_train` e `y_train` fornecem, respectivamente, as imagens e as classes corretas
que serão utilizadas para ajustar os pesos.

O treinamento pode ser entendido como a repetição destas operações:

1. Selecionar um lote de imagens de treino e seus rótulos.
2. Fazer as imagens passarem pelas camadas da rede e obter as probabilidades.
3. Comparar as probabilidades com os rótulos e calcular a perda do lote.
4. Calcular, por retropropagação, os gradientes da perda em relação aos parâmetros.
5. Utilizar o Adam para atualizar os pesos e vieses.
6. Repetir o processo para os demais lotes da época.
7. Ao final da época, calcular as métricas de validação sem atualizar os pesos.

Essas operações são realizadas pelo TensorFlow/Keras. O código do trabalho não
precisa implementar manualmente as derivadas e as atualizações.

### 7.2. `validation_data=(X_val, y_val)`

Esse argumento informa explicitamente os dados de validação definidos antes do
treinamento. O Keras calcula `val_loss` e `val_accuracy` ao final de cada época.

As imagens de validação não participam das atualizações dos pesos. Seus resultados
ajudam a observar se a melhora no treino também aparece em imagens que a rede
não utilizou para aprender.

Usar essa tupla mantém a divisão estratificada já criada. Não é necessário separar
novamente a validação dentro de `fit`.

O conjunto de teste não é fornecido nessa chamada. Assim, ele não orienta a análise
das épocas nem as decisões sobre a MLP nesta etapa.

### 7.3. `epochs=20`

Uma época corresponde a uma passagem pelo conjunto de treino. Com 20 épocas,
as mesmas 48.000 imagens são utilizadas em 20 passagens, enquanto os pesos vão
sendo atualizados. Isso não cria 20 conjuntos diferentes de imagens.

A execução termina ao completar a época 20. Não foram configurados callbacks de
parada antecipada ou restauração dos melhores pesos. Portanto, uma melhora ou
piora na validação não interrompe automaticamente o treinamento.

### 7.4. `batch_size=32`

O tamanho do lote define quantas imagens de treino são processadas em cada etapa
de atualização. Neste experimento:

```text
48.000 imagens ÷ 32 imagens por lote = 1.500 lotes por época
1.500 atualizações por época × 20 épocas = 30.000 atualizações
```

O registro `1500/1500` nas saídas de treinamento corresponde à conclusão desses
lotes em cada época. O tamanho do lote não é a quantidade de neurônios nem o
número de épocas.

Lotes permitem ajustar os pesos sem calcular todas as 48.000 imagens de uma só
vez em cada atualização. O tamanho 32 é uma escolha inicial do experimento, sem
comparação com outros tamanhos nesta etapa.

### 7.5. `verbose=2`

Esse parâmetro controla a apresentação do progresso. Com `verbose=2`, o Keras
mostra um resumo por época, incluindo as métricas de treino e validação.

Ele altera a quantidade de informações exibidas, não a arquitetura, a perda ou
o objetivo do aprendizado.

### 7.6. Como interpretar uma linha de treinamento

Uma das saídas reais foi:

```text
Epoch 20/20
1500/1500 - 3s - 2ms/step - accuracy: 0.9383 - loss: 0.1704 - val_accuracy: 0.8860 - val_loss: 0.3603
```

| Elemento da saída | Significado |
| --- | --- |
| `Epoch 20/20` | Última época das 20 configuradas. |
| `1500/1500` | Todos os lotes de treino da época foram processados. |
| `3s` e `2ms/step` | Tempos informados nessa execução; podem mudar em outro ambiente. |
| `accuracy: 0.9383` | Acurácia acumulada durante o treino da época, aproximadamente 93,83%. |
| `loss: 0.1704` | Perda acumulada durante o treino da época. |
| `val_accuracy: 0.8860` | Acurácia na validação ao final da época, aproximadamente 88,60%. |
| `val_loss: 0.3603` | Perda na validação ao final da época. |

Os tempos e números mostrados no log são arredondados. O CSV conserva mais casas
decimais das métricas.

### 7.7. Uma diferença importante entre métricas de treino e validação

As métricas de treino são acumuladas enquanto os pesos mudam de um lote para
outro. A validação é calculada depois da época, com os pesos então disponíveis e
sem atualizá-los durante essa avaliação.

Por isso, a acurácia de treino registrada não é exatamente uma nova avaliação
do modelo final da época em todas as imagens de treino. São medições úteis para
acompanhar o processo, mas feitas em condições diferentes.

Na primeira época, a validação atingiu 84,55% e o treino registrou 81,97%. Isso,
isoladamente, não indica um erro na divisão dos dados: a validação utiliza os
pesos após as atualizações da época, enquanto parte das previsões de treino foi
feita com pesos anteriores. A composição dos conjuntos também pode influenciar
a comparação.

## 8. Histórico, tabela e modelo salvo

### 8.1. `history_mlp` e `history_mlp.history`

`fit` retorna um objeto `History`, guardado em `history_mlp`. Seu atributo
`.history` é um dicionário com as métricas registradas ao longo das épocas.

| Chave | Conteúdo |
| --- | --- |
| `'loss'` | Uma perda de treino por época. |
| `'accuracy'` | Uma acurácia de treino por época. |
| `'val_loss'` | Uma perda de validação por época. |
| `'val_accuracy'` | Uma acurácia de validação por época. |

Como foram executadas 20 épocas, cada lista contém 20 valores. Essas listas
alimentam a tabela e os gráficos; os resultados não foram preenchidos manualmente
no histórico.

### 8.2. `pd.DataFrame(history_mlp.history)`

Essa chamada transforma o dicionário em uma tabela chamada `historico_mlp`.
Cada linha corresponde a uma época, e cada coluna corresponde a uma métrica.

`history_mlp` é o objeto retornado pelo treinamento. `historico_mlp` é a tabela
do pandas construída a partir dele. São variáveis relacionadas, mas com funções
e tipos diferentes.

### 8.3. `pd.RangeIndex`

A instrução:

```python
historico_mlp.index = pd.RangeIndex(1, len(historico_mlp) + 1, name='epoca')
```

define o índice da tabela como 1, 2, ..., 20 e dá a ele o nome `epoca`.
O limite final de `RangeIndex` não é incluído; por isso, o código usa
`len(historico_mlp) + 1`.

O objetivo é associar os resultados à numeração real das épocas. Sem esse ajuste,
o índice padrão do DataFrame começaria em 0.

### 8.4. `to_csv`

`historico_mlp.to_csv(RESULTADOS / 'historico_mlp.csv')` grava a tabela em
[resultados/historico_mlp.csv](../resultados/historico_mlp.csv).

O arquivo contém o índice `epoca` e as quatro métricas. Ele permite consultar os
números sem repetir o treinamento. Valores como `0.8859999775886536` refletem a
representação numérica de ponto flutuante e são apresentados arredondados como
88,60% na análise.

### 8.5. `model_mlp.save`

`model_mlp.save(MODELOS / 'mlp.keras')` salva o modelo no arquivo
[modelos/mlp.keras](../modelos/mlp.keras), incluindo sua arquitetura e os pesos
aprendidos, além das informações de treinamento suportadas pelo formato.

A chamada ocorre depois de `fit`, portanto esse arquivo corresponde à **época
20**. Ele não contém automaticamente os pesos da época de menor perda de
validação. Não foi configurado um mecanismo para salvar ou restaurar os pesos
da época 6.

Salvar o modelo não calcula novas métricas e não avalia o conjunto de teste.

### 8.6. `display(historico_mlp.iloc[[0, -1]])`

`iloc` seleciona linhas por posição:

- `0` seleciona a primeira linha, correspondente à época 1;
- `-1` seleciona a última linha, correspondente à época 20;
- `[[0, -1]]` mantém as duas linhas no formato de tabela.

`display` apresenta essa tabela no notebook. A operação apenas escolhe quais
linhas mostrar; todas as 20 épocas continuam presentes no CSV e nos gráficos.

### 8.7. `idxmin()`

A expressão:

```python
historico_mlp['val_loss'].idxmin()
```

retorna o rótulo do índice da linha que possui a menor perda de validação.
Como o índice foi definido com os números das épocas, o resultado desta execução
foi **6**.

`idxmin()` identifica uma linha da tabela. Ele não muda os pesos do modelo,
não volta o treinamento para uma época anterior e não substitui o arquivo salvo.

## 9. Gráficos: código completo

A terceira célula de código adapta a visualização `plot_metrics` da aula:

```python
# Adaptado de plot_metrics da aula, incluindo as curvas de validação.
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

ax[0].plot(historico_mlp.index, historico_mlp['loss'], label='Treino')
ax[0].plot(historico_mlp.index, historico_mlp['val_loss'], label='Validação')
ax[0].set_title('MLP — perda por época')
ax[0].set_xlabel('Época')
ax[0].set_ylabel('Loss (entropia cruzada)')
ax[0].grid(True)
ax[0].legend()

ax[1].plot(historico_mlp.index, historico_mlp['accuracy'], label='Treino')
ax[1].plot(historico_mlp.index, historico_mlp['val_accuracy'], label='Validação')
ax[1].set_title('MLP — acurácia por época')
ax[1].set_xlabel('Época')
ax[1].set_ylabel('Acurácia')
ax[1].set_ylim(0, 1.05)
ax[1].grid(True)
ax[1].legend()

fig.tight_layout()
fig.savefig(RESULTADOS / 'curvas_mlp.png', dpi=150, bbox_inches='tight')
plt.show()
```

## 10. Explicação de cada elemento dos gráficos

### 10.1. `plt.subplots(1, 2, figsize=(14, 5))`

A chamada cria uma figura com uma linha e duas colunas de gráficos:

- `fig` representa a figura completa;
- `ax[0]` é o gráfico da esquerda, usado para perda;
- `ax[1]` é o gráfico da direita, usado para acurácia;
- `figsize=(14, 5)` define largura e altura em polegadas.

Essa organização segue a visualização da aula e permite observar as duas métricas
lado a lado.

### 10.2. As quatro chamadas de `plot`

Em todas elas, `historico_mlp.index` fornece as épocas de 1 a 20 no eixo
horizontal. O segundo argumento escolhe os valores do eixo vertical:

| Gráfico | Coluna | Curva |
| --- | --- | --- |
| `ax[0]` | `'loss'` | Perda de treino. |
| `ax[0]` | `'val_loss'` | Perda de validação. |
| `ax[1]` | `'accuracy'` | Acurácia de treino. |
| `ax[1]` | `'val_accuracy'` | Acurácia de validação. |

Assim, são quatro curvas distribuídas em dois gráficos, cobrindo as duas métricas
nos dois conjuntos exigidos pelo enunciado.

`label='Treino'` e `label='Validação'` dão nomes às curvas para a legenda.
As linhas unem as medições por época; não representam medições adicionais entre
uma época e outra.

### 10.3. Títulos, eixos, grade e legenda

- `set_title(...)` informa o que cada gráfico apresenta.
- `set_xlabel('Época')` identifica o eixo horizontal.
- `set_ylabel(...)` identifica a métrica do eixo vertical.
- `grid(True)` acrescenta uma grade para facilitar a leitura.
- `legend()` exibe os nomes das curvas definidos em `label`.
- `set_ylim(0, 1.05)` fixa a escala vertical da acurácia e deixa uma margem acima
  de 1. A acurácia continua limitada a 1; a margem visual não significa que possam
  existir mais de 100% de acertos.

### 10.4. Ajuste, exportação e exibição

`fig.tight_layout()` ajusta os espaços entre os gráficos, títulos e rótulos.

`fig.savefig(...)` salva a figura em
[resultados/curvas_mlp.png](../resultados/curvas_mlp.png):

- `dpi=150` define a resolução de exportação;
- `bbox_inches='tight'` ajusta a área salva ao conteúdo da figura.

`plt.show()` exibe a figura no notebook. Sua saída gráfica foi preservada no
arquivo `.ipynb`, além da imagem externa.

Essas operações utilizam o histórico já obtido. Elas não repetem o treino nem
alteram o modelo.

## 11. Como ler as curvas

No gráfico de perda, valores menores indicam que a rede está atribuindo, em
conjunto, probabilidades melhores às classes verdadeiras. No gráfico de acurácia,
valores maiores indicam uma proporção maior de classes previstas corretamente.

É necessário observar treino e validação em conjunto:

| Comportamento observado | Interpretação possível |
| --- | --- |
| Perda diminui e acurácia melhora nos dois conjuntos | O aprendizado também está aparecendo na validação. |
| Treino melhora enquanto a perda de validação aumenta | Indício de sobreajuste. |
| Resultados fracos nos dois conjuntos e pouca melhora | Pode haver subajuste ou dificuldades de otimização; é necessário investigar. |
| Acurácia quase constante, mas perda muda | As probabilidades podem estar mudando sem alterar muitas classes previstas. |

Uma pequena oscilação isolada não determina o diagnóstico. A interpretação deve
considerar a tendência das curvas, a diferença entre os conjuntos e os valores
medidos.

A figura gerada nesta execução é apresentada abaixo. Ela depende da presença
do arquivo na pasta de resultados; também pode ser consultada nas saídas salvas
do notebook.

![Curvas reais de perda e acurácia da MLP, para treino e validação ao longo de 20 épocas.](../resultados/curvas_mlp.png)

## 12. Resultados reais e análise

### 12.1. Valores principais da execução

Os valores abaixo vêm de `resultados/historico_mlp.csv` e das saídas salvas no
notebook. Acurácias foram convertidas para porcentagens e arredondadas a duas
casas; perdas foram arredondadas a quatro casas.

| Época | Loss de treino | Acurácia de treino | Loss de validação | Acurácia de validação |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0,4992 | 81,97% | 0,4168 | 84,55% |
| 6 | 0,2799 | 89,71% | **0,3278** | 88,27% |
| 18 | 0,1795 | 93,45% | 0,3516 | **88,60%** |
| 20 | **0,1704** | **93,83%** | 0,3603 | **88,60%** |

A menor perda de validação ocorreu na época 6. A maior acurácia de validação,
88,60%, ocorreu nas épocas 18 e 20. Isso mostra que a época mais favorável depende
da métrica observada: menor perda e maior acurácia não precisam coincidir.

### 12.2. Evolução do aprendizado

No início, a rede melhorou nos dois conjuntos. Entre as épocas 1 e 6, a perda
de validação caiu de 0,4168 para 0,3278, enquanto a acurácia de validação passou
de 84,55% para 88,27%.

Até a época 20, o treino continuou melhorando: sua perda chegou a 0,1704 e a
acurácia a 93,83%. A validação apresentou ganhos bem menores de acurácia, e sua
perda aumentou em relação ao mínimo observado.

### 12.3. Evidências de sobreajuste

Depois da época 6, a tendência geral foi de queda da perda de treino e aumento
da perda de validação, embora existam oscilações entre épocas. Na última época,
há uma diferença de aproximadamente **5,23 pontos percentuais** entre as
acurácias de treino e validação:

```text
93,83% − 88,60% = 5,23 pontos percentuais
```

Esse comportamento sugere que parte da melhora no treino não se transferiu para
a validação, caracterizando indícios de sobreajuste. Ele não permite afirmar,
sozinho, que a rede apenas memorizou imagens ou que deixou de aprender padrões
úteis.

O padrão predominante não é o de subajuste: a rede consegue melhorar no treino,
mas os ganhos na validação ficam limitados.

### 12.4. Relação observada entre perda e acurácia

Entre as épocas 6 e 20:

- A acurácia de validação passou de 88,27% para 88,60%.
- A perda de validação passou de 0,3278 para 0,3603.

Houve uma pequena melhora na proporção de acertos e, ao mesmo tempo, uma piora
na perda. Portanto, não seria correto descrever o modelo como melhor em todos os
aspectos apenas porque a acurácia final foi maior.

As métricas agregadas são compatíveis, por exemplo, com erros mais confiantes:
algumas probabilidades muito baixas para classes verdadeiras podem elevar a perda,
mesmo que outras imagens passem a ser classificadas corretamente. Não foram
analisadas as probabilidades individuais para confirmar que esse mecanismo seja
a única explicação.

### 12.5. O desempenho é satisfatório?

Como resultado inicial para servir de comparação com a CNN, o desempenho é
satisfatório: a MLP classificou corretamente cerca de 88,60% das imagens de
validação.

Para contextualizar, um classificador que sempre escolhesse uma única classe
acertaria 10% nessa validação, pois cada uma das dez classes tem 1.200 imagens.
Essa comparação decorre da distribuição dos rótulos; não foi treinado outro modelo
para produzir esse valor de referência.

O enunciado não define uma acurácia mínima obrigatória. O resultado de 88,60%
também não demonstra que a configuração seja ótima. A subida da perda de
validação e a diferença para o treino precisam acompanhar a avaliação positiva.

### 12.6. O que se pode concluir sobre generalização

A validação fornece evidência de que a rede aprendeu padrões úteis para imagens
que não participaram das atualizações dos pesos. Entretanto, a análise foi feita
com uma única divisão de dados e uma única execução da configuração.

Não foram calculados resultados no teste para a MLP. Portanto, os 88,60% devem
ser identificados como **acurácia de validação**, e não como acurácia de teste ou
garantia de desempenho em qualquer conjunto de roupas.

A comparação com a CNN será feita depois de sua implementação e execução.
Não é possível afirmar, neste momento, que uma arquitetura superou a outra.

### 12.7. Modelo final e melhor época segundo a perda

Há três informações diferentes que precisam ser mantidas claras:

| Informação | Resultado desta execução |
| --- | --- |
| Época de menor perda de validação | 6 |
| Épocas de maior acurácia de validação | 18 e 20 |
| Época dos pesos em `modelos/mlp.keras` | 20 |

O código identifica a época de menor perda apenas para análise. Não houve
restauração de seus pesos. Não se deve atribuir ao modelo salvo a perda de
validação da época 6.

## 13. Configuração e arquivos do experimento

### 13.1. Configuração da MLP

| Elemento | Configuração |
| --- | --- |
| Dataset | Fashion-MNIST fornecido no trabalho. |
| Entrada | Imagens de 28 × 28 × 1, normalizadas para `[0, 1]`. |
| Divisão | 48.000 imagens de treino e 12.000 de validação estratificada. |
| Semente | 42. |
| Arquitetura | `Flatten → Dense(128, tanh) → Dense(10, softmax)`. |
| Parâmetros aprendidos da rede | 101.770. |
| Otimizador | Adam. |
| Taxa de aprendizagem | 0,001. |
| Perda | `sparse_categorical_crossentropy`. |
| Métrica adicional | `accuracy`. |
| Tamanho do lote | 32. |
| Épocas | 20. |
| Parada | Ao completar as 20 épocas. |
| Aumento de dados | Não utilizado na MLP. |
| Teste | Não utilizado nesta etapa. |

### 13.2. Ambiente registrado

O arquivo `resultados/ambiente.json`, gerado pela preparação do notebook,
registra as seguintes versões na execução documentada:

| Componente | Versão |
| --- | --- |
| Python | 3.13.14 |
| TensorFlow | 2.21.0 |
| NumPy | 2.4.6 |
| pandas | 3.0.3 |
| Matplotlib | 3.11.1 |
| scikit-learn | 1.9.0 |

A saída da preparação do ambiente não apresentou GPUs disponíveis. O treinamento
documentado foi realizado em CPU. As versões descrevem essa execução, sem impor
uma promessa de reprodução exata em outro ambiente.

### 13.3. Arquivos associados

| Arquivo | Conteúdo |
| --- | --- |
| `Trabalho_02_Redes_Neurais.ipynb` | Código, explicações, logs, tabela, gráficos e análise com as saídas salvas. |
| `resultados/historico_mlp.csv` | As quatro métricas das 20 épocas. |
| `resultados/curvas_mlp.png` | Os dois gráficos com as quatro curvas. |
| `modelos/mlp.keras` | Modelo com os pesos após a época 20. |
| `resultados/divisao_dados.npz` | Índices da divisão de treino e validação, gerados na preparação. |
| `resultados/ambiente.json` | Semente e versões registradas na preparação. |

As pastas de dados, resultados e modelos possuem arquivos ignorados pelo Git.
Assim, uma cópia obtida apenas pelo versionamento pode não incluir o CSV, a imagem
externa ou o modelo. As saídas salvas no notebook continuam sendo parte central
da entrega e permitem consultar os resultados documentados.

## 14. Como acompanhar a implementação no notebook

1. Abra `Trabalho_02_Redes_Neurais.ipynb` no ambiente Jupyter do projeto.
2. Leia e execute, se necessário, as etapas de ambiente, carregamento, divisão
   dos dados e normalização, na ordem em que aparecem.
3. Vá para a seção **4. MLP: implementação, treinamento e análise (item 1)**.
4. Leia o **Passo 1** e a célula de construção. O `summary()` permite conferir
   formatos e quantidades de parâmetros.
5. Leia o **Passo 2** e a célula de treinamento. Executá-la inicia um treinamento
   de 20 épocas e grava o histórico e o modelo nos caminhos descritos.
6. Leia o **Passo 3** e a célula de gráficos. Ela utiliza o histórico da execução
   disponível no kernel.
7. Leia o **Passo 4**, que contém a análise da execução registrada neste documento.

As células da MLP dependem das variáveis criadas nas anteriores. Reiniciar o kernel
apaga essas variáveis da memória, embora os arquivos salvos no disco continuem
existindo. Para executar as células de treinamento novamente, é necessário preparar
os dados e construir o modelo na ordem indicada.

Para apenas estudar o resultado já obtido, é possível ler as saídas salvas sem
executar novamente o treinamento. Uma nova execução pode produzir números
diferentes e substituir os arquivos gerados; nesse caso, a análise e esta
documentação devem ser conferidas em relação ao novo histórico.

## 15. Referências e registro de apoio

- **Profa. Nathalia da Cruz Alves:**
  [enunciado do Trabalho Prático 2](<Trabalho Prático 2 - Redes Neurais.pdf>) e
  [Aula 19 — Perceptron](../Aulas_Praticas/Aula_19_Perceptron_Estudante.ipynb).
  A construção da MLP e os gráficos foram adaptados do material da professora,
  preservando sua autoria.
- **Notebook da equipe:**
  [Trabalho_02_Redes_Neurais.ipynb](../Trabalho_02_Redes_Neurais.ipynb), seção 4,
  fonte do código e das saídas explicadas neste documento.
- **Slides da disciplina:**
  [Aula 19 — MLPs, Backpropagation e Gradient Descent](<../Conteudo/Aula 19 - MLPs, Backpropagation e Gradient Descent - INE5633.pdf>),
  disponibilizados como apoio conceitual do trabalho.
- **Fashion-MNIST:** origem, licença e referência do dataset descritas no
  [documento local dos dados](../dados/README.md).
- **OpenAI Codex:** apoio à adaptação da MLP da aula, execução do treinamento,
  elaboração dos gráficos, interpretação das métricas e redação desta documentação
  detalhada do item 1. O registro descreve o uso da ferramenta; não atribui aos
  estudantes contribuições que ainda não tenham sido informadas. A equipe deve
  revisar o conteúdo e compreender as decisões para apresentar o trabalho.
