# Item 3 — Experimentação e otimização de hiperparâmetros da CNN

Este documento explica a seção **5.1. Experimentação e otimização da CNN** do
[notebook da equipe](../Trabalho_02_Redes_Neurais.ipynb). Além do procedimento e
dos resultados, ele descreve como a equipe pode alterar as opções para realizar
novos testes sem modificar toda a implementação.

As explicações anteriores estão na
[documentação da MLP](ITEM_1_MLP.md) e na
[documentação da CNN com aumento de dados](ITEM_2_CNN.md). A preparação do dataset
continua sendo a mesma.

## 1. Requisito e material utilizado

O [enunciado](<Trabalho Prático 2 - Redes Neurais.pdf>) pede testar diferentes
combinações de hiperparâmetros da CNN, informar os valores ou intervalos, apresentar
uma tabela comparativa, indicar os melhores valores e analisar as diferenças
observadas.

A base de implementação é a
[Aula 23 — Treinamento de modelos](../Aulas_Praticas/Aula_23_Treinamento_de_modelos_ESTUDANTE.ipynb),
de autoria da **Profa. Nathalia da Cruz Alves**, especialmente:

- `build_model(hp)`, que constrói e compila uma CNN com valores escolhidos pelo
  Keras Tuner;
- `kt.RandomSearch`, que organiza as tentativas;
- `tuner.search`, que executa os treinamentos;
- a tabela construída a partir de `tuner.oracle.trials`;
- `get_best_models` e `get_best_hyperparameters`, que recuperam o modelo e a
  configuração escolhidos.

No exemplo da aula, a função usa duas convoluções, pooling em dois pontos e
BatchNormalization. Adaptamos essa função para preservar a sequência da CNN
do item 2: três convoluções, um pooling e uma camada densa. O Dropout após a
camada densa segue o exemplo da aula 23 e passa a ter sua taxa testada.

Essa escolha mantém o foco na variação dos hiperparâmetros definidos, sem mudar
também toda a topologia da CNN entre os itens 2 e 3. Não introduzimos uma classe
própria de tuner ou um sistema separado de experimentação.

Na aula, a quantidade de neurônios densos usa `hp.Int`, com intervalo e passo.
Aqui, utilizamos `hp.Choice` também nessa camada para que todas as opções fiquem
em listas no mesmo bloco editável. Os valores iniciais 32 e 64 pertencem ao
intervalo apresentado na aula.

## 2. Como a equipe pode fazer novos testes

### 2.1. O que editar

Na seção 5.1, localize a primeira célula de código, marcada com **EDITE AQUI**.
Ela contém:

| Elemento | O que a equipe pode alterar |
| --- | --- |
| `HIPERPARAMETROS` | As listas de valores que podem ser sorteados. |
| `MAX_TENTATIVAS` | O limite de combinações que a busca vai treinar. |
| `EPOCAS_POR_TENTATIVA` | Quantas épocas cada tentativa vai executar. |
| `BATCH_SIZE_BUSCA` | O tamanho do lote, igual para todas as tentativas dessa busca. |
| `NOME_BUSCA` | O nome que identifica e separa os arquivos do experimento. |

O tamanho do lote e as épocas são configurações editáveis, mas não são sorteados
como hiperparâmetros nesta implementação. Se quiser comparar dois tamanhos de
lote, faça buscas separadas, com nomes diferentes, e registre essa diferença na
análise.

### 2.2. Sequência para uma nova busca

1. Prepare os dados executando as seções 1, 2 e 3 do notebook, caso tenha
   reiniciado o kernel. O item 3 depende das variáveis de dados e caminhos
   criadas nessas seções.
2. Altere as listas ou o orçamento no bloco **EDITE AQUI**.
3. Escolha um novo nome, por exemplo `fashion_mnist_busca_02`.
4. Execute os passos 1 a 5 da seção 5.1, na ordem em que aparecem.
5. Confira a tabela e a configuração do modelo recuperado.
6. Atualize a análise do notebook com os resultados dessa nova execução.

A busca utiliza apenas os dados preparados: não depende de treinar novamente
a MLP ou a CNN do item 2. As saídas anteriores podem continuar salvas no notebook.
Nesta execução do item 3, a preparação e a busca foram executadas em um novo
kernel, preservando os treinamentos já documentados dos itens anteriores.

### 2.3. Exemplo de alteração

Para permitir uma camada densa maior, a equipe pode substituir a lista
`'dense_units': [32, 64]` por `'dense_units': [32, 64, 128]` e escolher outro
`NOME_BUSCA`. O código de `build_model` já consulta essa lista; não é necessário
reescrever a camada.

Esse é um exemplo de edição, não um experimento realizado. O documento registra
como resultados apenas as combinações efetivamente executadas.

Para testar uma única combinação, deixe cada lista com um valor, use
`MAX_TENTATIVAS = 1` e escolha outro `NOME_BUSCA`.

Também é possível planejar uma comparação variando apenas um hiperparâmetro.
O exemplo abaixo deixa apenas o dropout variar; **ele é uma sugestão de edição,
não uma busca já executada**:

```python
HIPERPARAMETROS = {
    'filters_1': [8],
    'filters_2': [16],
    'dense_units': [64],
    'dropout_rate': [0.0, 0.3],
    'learning_rate': [0.001],
}
MAX_TENTATIVAS = 2
EPOCAS_POR_TENTATIVA = 5
BATCH_SIZE_BUSCA = 32
NOME_BUSCA = 'comparacao_dropout_01'
```

Essa configuração permite comparar as duas opções de dropout com os demais
hiperparâmetros iguais. Uma única execução por opção ainda inclui variabilidade
de inicialização e treinamento; a comparação deve reconhecer esse limite.

### 2.4. Cuidados ao editar as opções

Mantenha os nomes das cinco chaves, porque `build_model` utiliza esses nomes
para localizar os valores. As listas devem ser não vazias e conter valores
compatíveis com cada camada:

- filtros e neurônios: inteiros positivos;
- dropout: números entre 0 e 1, excluindo 1;
- taxa de aprendizagem: números positivos;
- épocas, lote e tentativas: inteiros positivos.

Use tipos consistentes dentro de cada lista. No dropout, por exemplo, utilize
`0.0` junto com `0.3`, como no bloco inicial.

Adicionar opções não garante que todas serão treinadas: o Random Search continua
limitado por `MAX_TENTATIVAS`. Aumentar filtros, neurônios, épocas ou tentativas
pode elevar o custo de processamento e memória.

## 3. Configuração inicial e justificativas

O espaço inicial tem cinco escolhas com duas opções cada:

| Hiperparâmetro | Valores permitidos | Finalidade |
| --- | --- | --- |
| `filters_1` | 8, 16 | Comparar duas capacidades na primeira convolução. |
| `filters_2` | 16, 32 | Comparar duas capacidades na segunda convolução. |
| `dense_units` | 32, 64 | Variar o tamanho da camada densa. |
| `dropout_rate` | 0,0; 0,3 | Comparar ausência de descarte e descarte de 30% das ativações. |
| `learning_rate` | 0,001; 0,0001 | Comparar duas escalas de atualização do Adam. |

O total possível nesse espaço é:

```text
2 × 2 × 2 × 2 × 2 = 32 combinações
```

O orçamento inicial é de **cinco tentativas, com cinco épocas por tentativa**,
uma execução por combinação e lotes de 32. Esse orçamento curto segue a estratégia
da aula. Ele permite comparar algumas combinações, mas não cobre todo o espaço
nem garante que cada configuração tenha treinado até estabilizar.

As escolhas de filtros, neurônios e taxas são limitadas para manter o experimento
compreensível e executável em CPU. As taxas do Adam fazem parte das opções
apresentadas na aula. Dropout zero inclui uma configuração sem essa regularização,
enquanto 0,3 permite observar uma intensidade moderada.

### 3.1. Elementos que ficam fixos

Todas as tentativas utilizam:

- imagens de 28 × 28 × 1, normalizadas para `[0, 1]`;
- a mesma divisão de 48.000 imagens de treino e 12.000 de validação;
- aumento de dados com rotação 0,05, translação 0,1 e zoom 0,1;
- três camadas convolucionais de kernel 3 × 3 e ativação ReLU;
- 32 filtros na terceira convolução;
- um `MaxPooling2D` de 2 × 2, seguido de `Flatten`;
- saída de dez neurônios com `softmax`;
- Adam, entropia cruzada para rótulos inteiros e acurácia.

O número de camadas, os kernels e os fatores de aumento não são sorteados nesta
busca. O enunciado apresenta exemplos de hiperparâmetros; não exige variar todos
eles simultaneamente.

## 4. Configuração: código e arquivos de identificação

Esta é a primeira célula da implementação:

```python
import keras_tuner as kt

# EDITE AQUI: as listas contêm os valores que o Random Search pode sortear.
HIPERPARAMETROS = {
    'filters_1': [8, 16],
    'filters_2': [16, 32],
    'dense_units': [32, 64],
    'dropout_rate': [0.0, 0.3],
    'learning_rate': [0.001, 0.0001],
}
MAX_TENTATIVAS = 5
EPOCAS_POR_TENTATIVA = 5
BATCH_SIZE_BUSCA = 32
NOME_BUSCA = 'fashion_mnist_busca_01'  # Use outro nome para uma nova configuração.

PASTA_BUSCAS = RESULTADOS / 'buscas_cnn'
PASTA_BUSCA = PASTA_BUSCAS / NOME_BUSCA
configuracao_busca = {
    'hiperparametros': HIPERPARAMETROS,
    'max_tentativas': MAX_TENTATIVAS,
    'epocas_por_tentativa': EPOCAS_POR_TENTATIVA,
    'batch_size': BATCH_SIZE_BUSCA,
    'seed': SEED,
    'objetivo': 'val_accuracy',
    'arquitetura': 'CNN do item 2 com Dropout após Dense',
}
arquivo_configuracao = PASTA_BUSCA / 'configuracao_busca.json'
if arquivo_configuracao.exists():
    assert json.loads(arquivo_configuracao.read_text()) == configuracao_busca, (
        'Configuração diferente da busca salva. Escolha um novo NOME_BUSCA.'
    )
PASTA_BUSCA.mkdir(parents=True, exist_ok=True)
arquivo_configuracao.write_text(
    json.dumps(configuracao_busca, indent=2, ensure_ascii=False), encoding='utf-8',
)

print('Keras Tuner:', kt.__version__)
display(pd.DataFrame([
    {'hiperparametro': nome, 'valores': valores}
    for nome, valores in HIPERPARAMETROS.items()
]))
print(f'Busca: {NOME_BUSCA} | até {MAX_TENTATIVAS} tentativas')
print(f'Épocas por tentativa: {EPOCAS_POR_TENTATIVA} | lote: {BATCH_SIZE_BUSCA}')
```

### 4.1. Importação e dicionário

`import keras_tuner as kt` utiliza o mesmo apelido da aula. O pacote está previsto
nas dependências do projeto; a célula registra a versão instalada na saída.

`HIPERPARAMETROS` agrupa as cinco listas em um único lugar. As chaves seguem os
nomes usados na aula, o que facilita relacionar a configuração com `hp.Choice`.

### 4.2. Orçamento e nome

As três variáveis numéricas seguintes controlam tentativas, épocas e lote.
`NOME_BUSCA` identifica a pasta do experimento. O nome inicial é
`fashion_mnist_busca_01`.

`PASTA_BUSCAS` aponta para `resultados/buscas_cnn/`. `PASTA_BUSCA` acrescenta o
nome do experimento a esse caminho. Esses objetos são `Path`; o operador `/`
junta partes de um caminho.

### 4.3. Registro e conferência de configuração

`configuracao_busca` reúne as opções, o orçamento, a semente e o objetivo em um
dicionário que será salvo como `configuracao_busca.json` na pasta do experimento.

Se esse arquivo já existir, a célula compara seu conteúdo com a configuração
atual. Se forem diferentes, uma mensagem pede um novo `NOME_BUSCA`. Isso impede
misturar silenciosamente resultados salvos com outras listas ou outro orçamento.

A conferência cobre os campos registrados nesse dicionário. Se a equipe alterar
manualmente a arquitetura dentro de `build_model`, também deve criar um novo
nome e registrar a mudança; o código não compara automaticamente o texto da função.

`mkdir(parents=True, exist_ok=True)` cria a pasta, e `write_text` grava o JSON.
As chamadas de `display` e `print` mostram as opções e o orçamento efetivamente
definidos antes do treinamento.

## 5. Função de construção da CNN

A segunda célula define a função utilizada pelo tuner:

```python
# Adaptado de build_model(hp) da Aula Prática 23 e da CNN do item 2.
def build_model(hp):
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        keras.Sequential([
            keras.layers.RandomRotation(0.05),
            keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
            keras.layers.RandomZoom(0.1),
        ], name='aumento_dados'),
        keras.layers.Conv2D(
            filters=hp.Choice('filters_1', HIPERPARAMETROS['filters_1']),
            kernel_size=(3, 3), activation='relu',
        ),
        keras.layers.Conv2D(
            filters=hp.Choice('filters_2', HIPERPARAMETROS['filters_2']),
            kernel_size=(3, 3), activation='relu',
        ),
        keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(
            units=hp.Choice('dense_units', HIPERPARAMETROS['dense_units']),
            activation='relu',
        ),
        keras.layers.Dropout(
            rate=hp.Choice('dropout_rate', HIPERPARAMETROS['dropout_rate']),
        ),
        keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', HIPERPARAMETROS['learning_rate']),
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model
```

### 5.1. O que é `hp`

`hp` é um objeto de hiperparâmetros fornecido pelo Keras Tuner. A expressão:

```python
hp.Choice('filters_1', HIPERPARAMETROS['filters_1'])
```

associa o nome `filters_1` à lista de opções e devolve o valor escolhido para
a construção atual. Os demais `hp.Choice` funcionam da mesma forma.

O nome identifica a escolha nos registros do tuner e aparece na tabela final.
Por isso, alterar o nome de uma chave exige também conferir os pontos da função
que a utilizam.

### 5.2. Um modelo novo por tentativa

A função cria um novo `Sequential`, novas camadas e um novo bloco de aumento de
dados. Ela não reutiliza `model_cnn` nem seus pesos do item 2. O objetivo é comparar
treinamentos das configurações propostas, e não continuar o mesmo treinamento
com mudanças de estrutura.

Os filtros das duas primeiras convoluções, a quantidade de neurônios densos,
o dropout e a taxa do Adam vêm de `hp.Choice`. A terceira convolução permanece
com 32 filtros.

As dimensões espaciais continuam seguindo o caminho da CNN anterior:

```text
28 × 28 → 26 × 26 → 24 × 24 → 22 × 22 → pooling → 11 × 11
```

Como a terceira convolução tem 32 filtros, `Flatten` produz 3.872 valores em
todas as tentativas. A quantidade de parâmetros varia com os filtros anteriores
e, principalmente, com a quantidade de neurônios da camada densa.

### 5.3. Dropout

`Dropout` desativa aleatoriamente uma fração das ativações durante o treino.
Com taxa 0,3, a fração de descarte configurada é de 30%; isso não significa
remover permanentemente 30% dos neurônios ou dos pesos.

Na avaliação, a camada não faz esse descarte aleatório. A taxa zero não descarta
ativações. A técnica pode reduzir dependências específicas entre unidades, mas
também dificultar o aprendizado, especialmente com poucas épocas. Seu efeito
precisa ser discutido a partir dos resultados efetivamente observados.

### 5.4. Compilação

O Adam utiliza a taxa sorteada em `learning_rate`. A função de perda continua
sendo `sparse_categorical_crossentropy`, pois os rótulos são inteiros de 0 a 9.
A acurácia continua medindo a fração de classes previstas corretamente.

`return model` entrega ao tuner um modelo já construído e compilado. A função
não chama `fit`: o treinamento é organizado por `tuner.search` na próxima célula.

## 6. Criação e execução do Random Search

A terceira célula organiza os experimentos:

```python
keras.utils.set_random_seed(SEED)
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=MAX_TENTATIVAS,
    executions_per_trial=1,
    seed=SEED,
    directory=str(PASTA_BUSCAS),
    project_name=NOME_BUSCA,
    overwrite=False,
)

tuner.search(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCAS_POR_TENTATIVA,
    batch_size=BATCH_SIZE_BUSCA,
    verbose=2,
)
```

### 6.1. Argumentos de `kt.RandomSearch`

| Argumento | Significado |
| --- | --- |
| `build_model` | Função chamada para criar as CNNs com as opções selecionadas. |
| `objective='val_accuracy'` | Escolher pela maior acurácia de validação observada. |
| `max_trials=MAX_TENTATIVAS` | Limitar a quantidade de combinações treinadas. |
| `executions_per_trial=1` | Fazer um treinamento por combinação. |
| `seed=SEED` | Controlar o sorteio das combinações. |
| `directory` | Pasta principal dos experimentos do tuner. |
| `project_name` | Nome desta busca dentro da pasta principal. |
| `overwrite=False` | Reutilizar os registros existentes em vez de apagá-los. |

O Random Search amostra combinações do espaço informado. Ele não percorre
necessariamente todas as opções nem promete encontrar o melhor modelo possível.
O resultado é o melhor entre as tentativas que efetivamente concluiu.

### 6.2. Aleatoriedade

`keras.utils.set_random_seed(SEED)` configura as fontes usuais de aleatoriedade
antes da busca. O `seed` do tuner controla seu sorteio de hiperparâmetros.

Sementes ajudam a controlar o experimento, mas não garantem pesos e métricas
idênticos em todos os ambientes. Também não se deve prometer que interromper e
reabrir uma busca reproduzirá exatamente uma execução contínua.

Com uma execução por combinação, as diferenças observadas também incluem a
variabilidade de inicialização e treinamento. Não estimamos essa variabilidade
com repetições nesta busca inicial.

### 6.3. `tuner.search`

A chamada recebe `X_train` e `y_train` para atualização dos pesos e
`validation_data=(X_val, y_val)` para comparação e seleção. O teste não é passado
para o tuner.

O aumento de dados e o dropout fazem parte da rede e são ativados apenas no
treino. A validação utiliza as imagens normalizadas, sem transformações ou
descarte aleatório de ativações.

Cada tentativa concluída executa as épocas configuradas; não foi utilizado
`EarlyStopping`. Entretanto, o checkpoint selecionado pode corresponder a uma
época anterior à última, caso tenha apresentado maior `val_accuracy`.

`verbose=2` mostra as métricas por época. O tuner também informa os hiperparâmetros
de cada tentativa e o melhor resultado obtido até aquele momento. Essas saídas
reais ficam salvas no notebook.

### 6.4. Reabrir uma busca existente

Com o mesmo nome e a mesma configuração, o tuner pode aproveitar tentativas
concluídas e continuar o trabalho restante. Se todas já tiverem terminado, uma
nova chamada pode apenas recarregar os resultados.

Isso não representa uma nova repetição independente da busca. Para executar
outro experimento, utilize outro nome. Não há garantia de retomada no último
lote ou na última época de uma tentativa interrompida.

## 7. Tabela dos experimentos

A quarta célula adapta a tabela da aula:

```python
# Adaptado da tabela de trials da Aula Prática 23.
resultados_busca = []
for trial in tuner.oracle.trials.values():
    linha = {
        'tentativa': trial.trial_id,
        'status': trial.status,
        **trial.hyperparameters.values,
        'camadas_conv': 3,
        'filters_3': 32,
        'batch_size': BATCH_SIZE_BUSCA,
        'epocas_executadas': EPOCAS_POR_TENTATIVA if trial.status == 'COMPLETED' else None,
    }
    if trial.score is not None:
        linha['melhor_epoca'] = int(trial.best_step) + 1
        for metrica in ('loss', 'accuracy', 'val_loss', 'val_accuracy'):
            linha[metrica] = trial.metrics.get_best_value(metrica)
    resultados_busca.append(linha)

experimentos_cnn = pd.DataFrame(resultados_busca).sort_values(
    'val_accuracy', ascending=False, na_position='last',
).reset_index(drop=True)
experimentos_cnn.insert(0, 'posicao', range(1, len(experimentos_cnn) + 1))
display(experimentos_cnn)
experimentos_cnn.to_csv(RESULTADOS / 'experimentos_cnn.csv', index=False)
experimentos_cnn.to_csv(PASTA_BUSCA / 'experimentos_cnn.csv', index=False)
```

### 7.1. De onde vêm os valores

`tuner.oracle.trials` guarda os registros das tentativas. O laço percorre esses
registros e cria um dicionário por tentativa.

`trial.trial_id` identifica a tentativa. A numeração interna começa em zero;
assim, a tentativa de identificador `0` corresponde ao primeiro treinamento
apresentado pelo tuner como Trial 1.

`trial.hyperparameters.values` fornece os valores efetivamente escolhidos.
O trecho `**trial.hyperparameters.values` incorpora esses campos à linha da tabela.

`trial.status` informa se a tentativa foi concluída. Para tentativas com um
resultado válido, `trial.score` contém o resultado do objetivo.

### 7.2. Qual época é representada pelas métricas

`trial.best_step` registra a posição da melhor época segundo o objetivo. Como
essa posição começa em zero, somamos 1 para obter a época mostrada à equipe.

Na implementação padrão utilizada do Keras Tuner, o retorno de `fit` é convertido
em um conjunto de métricas da época de maior `val_accuracy`. Esse conjunto é
registrado na tentativa. Portanto, as quatro métricas consultadas com
`trial.metrics.get_best_value(...)` pertencem à mesma época selecionada.

Em particular, a coluna `val_loss` não deve ser interpretada como a menor perda
de todas as épocas da tentativa. Ela é a perda na época selecionada pela acurácia.
O nome do método `get_best_value` não altera a regra de seleção desta busca.

Esse cuidado evita combinar em uma linha a acurácia de uma época com a perda
de outra e apresentar essa mistura como um único resultado do modelo.

### 7.3. Colunas da tabela

| Coluna | Conteúdo |
| --- | --- |
| `posicao` | Ordem da tabela pela acurácia de validação. |
| `tentativa` | Identificador do tuner. |
| `status` | Estado registrado para a tentativa. |
| `filters_1`, `filters_2` | Filtros sorteados nas primeiras convoluções. |
| `dense_units` | Quantidade sorteada de neurônios densos. |
| `dropout_rate` | Taxa sorteada de dropout. |
| `learning_rate` | Taxa sorteada do Adam. |
| `camadas_conv` | Três camadas convolucionais, fixas. |
| `filters_3` | 32 filtros fixos na terceira convolução. |
| `batch_size` | Tamanho do lote desta busca. |
| `epocas_executadas` | Orçamento completo para tentativas concluídas; não inferido para falhas. |
| `melhor_epoca` | Época escolhida pela acurácia de validação. |
| `loss`, `accuracy` | Métricas de treino dessa época. |
| `val_loss`, `val_accuracy` | Métricas de validação dessa mesma época. |

O número de épocas executadas é conhecido nas tentativas concluídas desta
implementação porque o treino tem orçamento fixo e não usa parada antecipada.
A melhor época pode ser menor que esse número.

### 7.4. Ordenação, exibição e exportação

`pd.DataFrame(resultados_busca)` organiza os dicionários em uma tabela.
`sort_values(..., ascending=False)` ordena pela maior acurácia de validação;
valores ausentes ficam no fim. `reset_index(drop=True)` reorganiza o índice, e
`insert` acrescenta uma coluna de posições iniciando em 1.

`display(experimentos_cnn)` apresenta a tabela no notebook. As chamadas de
`to_csv` geram duas cópias:

- `resultados/experimentos_cnn.csv`: tabela da busca exportada mais recentemente;
- `resultados/buscas_cnn/NOME_BUSCA/experimentos_cnn.csv`: tabela preservada junto
  aos registros daquele experimento.

A ordem da tabela serve para leitura. O modelo vencedor é recuperado pelos
métodos do próprio tuner. Se houver empate de acurácia, não atribuímos uma regra
adicional de desempate pela perda.

## 8. Recuperação do melhor modelo

A quinta célula recupera os resultados selecionados:

```python
melhor_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
melhores_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
melhor_modelo = tuner.get_best_models(num_models=1)[0]

melhor_configuracao = {
    'nome_busca': NOME_BUSCA,
    'tentativa': melhor_trial.trial_id,
    'melhor_epoca': int(melhor_trial.best_step) + 1,
    'hiperparametros': melhores_hp.values,
    'val_accuracy': float(melhor_trial.score),
    'val_loss': float(melhor_trial.metrics.get_best_value('val_loss')),
    'configuracao_busca': configuracao_busca,
}
melhor_modelo.save(MODELOS / 'melhor_cnn.keras')
(MODELOS / 'melhor_cnn_configuracao.json').write_text(
    json.dumps(melhor_configuracao, indent=2, ensure_ascii=False), encoding='utf-8',
)
print(json.dumps(melhor_configuracao, indent=2, ensure_ascii=False))
melhor_modelo.summary()
```

### 8.1. Tentativa, hiperparâmetros e checkpoint

`tuner.oracle.get_best_trials(1)` identifica a tentativa escolhida pelo objetivo.
`get_best_hyperparameters(1)` recupera seus hiperparâmetros.

`get_best_models(1)` constrói a estrutura correspondente e carrega os pesos do
checkpoint escolhido durante a busca. Não se trata de um treinamento novo.
Por isso, não substituímos os pesos selecionados por uma nova execução e depois
atribuímos a ela as métricas antigas.

Essa distinção é especialmente relevante quando a melhor época não é a última.
Os pesos para avaliação final devem corresponder ao checkpoint selecionado,
não apenas a uma arquitetura com os mesmos hiperparâmetros.

### 8.2. Arquivos exportados

`melhor_modelo.save(...)` salva o modelo em `modelos/melhor_cnn.keras`.
O JSON `modelos/melhor_cnn_configuracao.json` registra:

- nome da busca;
- identificador da tentativa;
- melhor época;
- hiperparâmetros escolhidos;
- acurácia e perda de validação dessa época;
- configuração e orçamento da busca.

Os arquivos com esses nomes fixos representam a última busca exportada. Os
registros e checkpoints das buscas anteriores permanecem nas pastas nomeadas do
tuner. `melhor_modelo.summary()` mostra a arquitetura recuperada.

O próximo item poderá utilizar o objeto `melhor_modelo` ou carregar o arquivo
salvo para avaliar o teste. Nenhuma avaliação do teste é realizada nesta etapa.

## 9. Resultados reais e análise

**Experimentos realizados.** A busca `fashion_mnist_busca_01` concluiu cinco
tentativas, cada uma com cinco épocas e lotes de 32, sem utilizar o teste.
O espaço continha 32 combinações possíveis. A tabela resumida abaixo está
ordenada por acurácia de validação; a tabela completa, nas saídas e no CSV,
também registra status, campos fixos e métricas de treino da época selecionada.
Os identificadores começam em zero: a tentativa `0` foi a primeira executada.

| Tentativa | Filtros 1 | Filtros 2 | Neurônios | Dropout | Learning rate | Melhor época | Acurácia val. | Loss val. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 8 | 32 | 64 | 0,3 | 0,0010 | 5 | 86,01% | 0,4061 |
| 1 | 8 | 32 | 32 | 0,0 | 0,0010 | 4 | 85,07% | 0,4131 |
| 3 | 16 | 16 | 32 | 0,0 | 0,0010 | 4 | 84,93% | 0,4337 |
| 2 | 16 | 32 | 64 | 0,0 | 0,0001 | 5 | 79,37% | 0,5523 |
| 0 | 16 | 16 | 32 | 0,0 | 0,0001 | 5 | 78,57% | 0,5641 |

**Melhores valores encontrados.** A tentativa `4` foi escolhida
pela maior acurácia de validação, na época **5**:

- Primeira convolução: **8 filtros**.
- Segunda convolução: **32 filtros**.
- Terceira convolução: **32 filtros**, fixos em todas as tentativas.
- Camada densa: **64 neurônios**.
- Dropout: **0,3**.
- Taxa de aprendizagem: **0,0010**.
- Acurácia de validação: **86,01%**.
- Loss de validação: **0,4061**.

O arquivo `modelos/melhor_cnn.keras` foi recarregado e conferido nas mesmas imagens
de validação, reproduzindo as métricas do checkpoint selecionado. O modelo tem
260.186 parâmetros treináveis. O teste não participou dessa conferência.

**Taxa de aprendizagem: diferença mais marcante observada.** As tentativas `0`
e `3` usaram os mesmos filtros, 16 e 16, os mesmos 32 neurônios densos e dropout
zero. Mudou a taxa do Adam: 0,0001 e 0,001, respectivamente. A acurácia de
validação passou de 78,57% para
84,93%, uma diferença de aproximadamente
**6,35 pontos percentuais**,
calculada antes do arredondamento da tabela.

Esse resultado é compatível com aprendizado mais rápido ao utilizar 0,001 neste
orçamento curto: atualizações menores podem exigir mais épocas para atingir um
resultado semelhante. Ainda assim, foi uma execução por configuração, com
variabilidade de inicialização e treinamento; não demonstramos que essa taxa
será superior em todos os orçamentos ou arquiteturas.

**Combinações de filtros.** As tentativas `1` e `3` utilizaram a mesma taxa 0,001,
32 neurônios e dropout zero, mas filtros 8/32 e 16/16. Obtiveram, respectivamente,
85,07% e 84,93%
de validação, diferença de apenas
0,14 ponto percentual.
As duas camadas mudaram juntas; essa comparação não determina o efeito isolado
dos filtros de cada camada. Os resultados próximos também não sustentam a ideia
de que simplesmente aumentar filtros em uma camada garante maior acurácia.

**Neurônios e dropout.** A tentativa `4` manteve os filtros 8/32 e a taxa 0,001
da tentativa `1`, mas aumentou a camada densa de 32 para 64 neurônios e passou
o dropout de zero para 0,3. A acurácia subiu de
85,07% para 86,01%,
aproximadamente 0,94
ponto percentual. Uma camada maior pode aumentar a capacidade de representação,
e o dropout pode reduzir dependências entre ativações; porém, ambos mudaram
juntos. Não podemos atribuir a melhoria apenas ao dropout ou apenas aos neurônios.

Com taxa 0,0001, as tentativas `0` e `2` também mudaram simultaneamente filtros
da segunda convolução e neurônios densos. Seus resultados foram próximos,
78,57% e 79,37%.
Nesta amostra, a diferença associada ao par de taxas foi maior que essas
diferenças entre combinações de capacidade; isso descreve os cinco testes,
sem estabelecer uma ordenação universal de importância dos hiperparâmetros.

**Métricas da mesma época.** A tentativa `1` foi selecionada na época 4, com
acurácia de validação de 85,07% e loss
0,4131. Na época 5, a loss caiu para aproximadamente
0,4123, mas a acurácia caiu para 84,86%. Como o objetivo é a acurácia, a tabela
registra a época 4 e sua loss correspondente. Não combinamos métricas de épocas
diferentes para apresentar um resultado artificialmente melhor.

**Comparação com o item 2 e limites.** A CNN anterior atingiu 88,63% de validação
após 20 épocas, enquanto a melhor desta busca atingiu
86,01% em cinco épocas. A busca curta não demonstrou
melhoria em relação àquela execução. Os orçamentos são diferentes e não permitem
atribuir a diferença apenas aos hiperparâmetros. A configuração vencedora ainda
melhorou na última época, o que também limita conclusões sobre convergência.

As métricas de treino da melhor tentativa foram 81,94%
de acurácia e loss 0,4835, com aumento de dados e dropout ativos.
A validação usa imagens sem essas transformações e sem descarte de ativações;
portanto, sua acurácia maior que a de treino não é, isoladamente, sinal de erro.

O resultado representa o melhor entre as combinações realmente testadas, com
uma execução por combinação e uma única divisão. Para novos testes, a equipe
pode editar as listas ou o orçamento e escolher outro `NOME_BUSCA`, preservando
este experimento. Se ampliar as épocas para comparar com o item 2, os novos
resultados deverão ser executados e analisados, sem reutilizar as conclusões atuais.
A avaliação independente no conjunto de teste permanece para o item 4.

## 10. Limites da comparação

São cinco combinações de um espaço inicial com 32 possibilidades e apenas uma
execução por combinação. O melhor resultado encontrado não é um ótimo global
nem uma estimativa da variabilidade entre várias execuções.

As comparações dentro da busca utilizam os mesmos dados e orçamento. Quando
vários hiperparâmetros diferem entre duas tentativas, não é possível atribuir
isoladamente a um deles toda a diferença de desempenho.

A busca inicial tem cinco épocas por tentativa, enquanto a CNN do item 2 teve
20 épocas. Comparar seus valores finais ajuda a contextualizar os experimentos,
mas não é uma comparação com orçamento de treinamento igual. Uma configuração
que aprende mais devagar pode ser desfavorecida pelo orçamento curto.

Não foram feitas novas escolhas com base no teste. A avaliação final em dados
de teste permanece para o item 4, depois de encerradas as escolhas de configuração.

## 11. Arquivos e preservação dos experimentos

| Caminho | Conteúdo |
| --- | --- |
| `Trabalho_02_Redes_Neurais.ipynb` | Configuração, função de construção, execução, tabela, melhor modelo e análise, com saídas salvas. |
| `resultados/buscas_cnn/NOME_BUSCA/configuracao_busca.json` | Opções e orçamento registrados. |
| `resultados/buscas_cnn/NOME_BUSCA/` | Estado do tuner, tentativas e checkpoints. |
| `resultados/buscas_cnn/NOME_BUSCA/experimentos_cnn.csv` | Tabela daquele experimento. |
| `resultados/experimentos_cnn.csv` | Tabela da última busca exportada. |
| `modelos/melhor_cnn.keras` | Modelo selecionado da última busca exportada. |
| `modelos/melhor_cnn_configuracao.json` | Identificação e métricas desse modelo. |

As pastas de dados, resultados e modelos têm arquivos ignorados pelo Git. Uma
cópia obtida apenas pelo versionamento pode não incluir os checkpoints e CSVs.
As saídas salvas no notebook permitem consultar os resultados documentados;
para reabrir uma busca sem refazer seus treinamentos, é necessário preservar
também sua pasta do tuner.

## 12. Referências e registro de apoio

- **Profa. Nathalia da Cruz Alves:**
  [enunciado](<Trabalho Prático 2 - Redes Neurais.pdf>),
  [Aula 23 — Treinamento de modelos](../Aulas_Praticas/Aula_23_Treinamento_de_modelos_ESTUDANTE.ipynb)
  e [Aula 21 — CNN](../Aulas_Praticas/Aula_21_CNN_MNIST_ESTUDANTE.ipynb).
  O código foi adaptado desses materiais, preservando a atribuição de autoria.
- **Notebook da equipe:**
  [Trabalho_02_Redes_Neurais.ipynb](../Trabalho_02_Redes_Neurais.ipynb), seção 5.1,
  fonte da implementação e das saídas explicadas neste documento.
- **OpenAI Codex:** apoio à adaptação da função de construção e do Random Search,
  organização do bloco editável, execução dos experimentos, tabela, recuperação
  do modelo, análise e redação desta documentação. A equipe deve revisar e
  compreender as escolhas e os resultados antes de apresentar o trabalho.
