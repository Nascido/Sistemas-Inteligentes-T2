# Orientações para agentes — Trabalho 02 de Sistemas Inteligentes

Estas instruções se aplicam a todo o repositório e devem orientar qualquer agente que trabalhe nele.

## Objetivo e princípio central

Desenvolver o trabalho de classificação de imagens do **Fashion-MNIST**, com 10 classes de vestuário, seguindo estritamente o enunciado e utilizando fielmente os códigos das aulas práticas como base.

**Priorizar simplicidade, clareza e fidelidade ao material fornecido.** O código deve ser fácil de entender e de explicar posteriormente na apresentação. Implementar somente o que for necessário para cumprir os requisitos do trabalho, sem overengineering, funcionalidades extras ou complexidade desnecessária.

## Fontes que devem orientar o desenvolvimento

- **Enunciado:** `Trabalho Prático 2 - Redes Neurais.pdf` e a descrição fornecida pelo usuário. Conferir o enunciado antes de desenvolver ou alterar uma etapa.
- **MLP:** `Aulas_Praticas/Aula_20_Perceptron_Estudante.ipynb`, especialmente a implementação de MLP.
- **CNN e aumento de dados:** `Aulas_Praticas/Aula_21_CNN_MNIST_ESTUDANTE.ipynb`.
- **Otimização de hiperparâmetros e avaliação:** `Aulas_Praticas/Aula_23_Treinamento_de_modelos_ESTUDANTE.ipynb`.
- **Apoio conceitual:** slides em `Conteudo/`.
- **Dados:** Fashion-MNIST em `dados/`, observando a organização e as instruções locais.
- **Notebook da equipe:** `Trabalho_02_Redes_Neurais.ipynb`. Continuar seu desenvolvimento, aproveitando o que já estiver correto.

Ler o código da aula correspondente antes de implementar cada etapa. Os notebooks das aulas são a base de implementação, não apenas referências conceituais. Preservar os materiais e sua autoria.

## Como escrever e adaptar o código

- Manter a estrutura, as bibliotecas e a forma de construir, compilar, treinar e avaliar os modelos usadas nas aulas, com TensorFlow/Keras.
- Adaptar somente o necessário ao Fashion-MNIST e aos requisitos do enunciado: entradas, número de classes, saída, função de perda, validação, gráficos e experimentos solicitados. Não reproduzir configurações de problemas binários ou XOR quando incompatíveis com as 10 classes.
- Na otimização, usar como base a função de construção do modelo e o `keras_tuner.RandomSearch` da aula 23. Definir uma busca limitada e compreensível, suficiente para comparar combinações e atender ao enunciado.
- Preferir células curtas, código direto, nomes claros e funções simples quando houver necessidade real de reaproveitamento. Evitar classes próprias, camadas de abstração, configurações genéricas e fragmentação em módulos sem necessidade.
- Não adicionar bibliotecas, arquiteturas, técnicas, métricas, experimentos ou ferramentas que não sejam necessários ao trabalho. A presença de uma atividade extra em uma aula não a torna obrigatória no projeto.
- Não criar interfaces, serviços, sistemas de experimentação ou outras funcionalidades adicionais. Não fazer refatorações ou mudanças de infraestrutura sem relação com a tarefa solicitada.
- Explicar brevemente, em texto no notebook, as adaptações relevantes em relação às aulas e sua necessidade para o trabalho.
- Escrever explicações em português, com linguagem simples e suficiente para a equipe compreender e apresentar cada decisão.

## Requisitos das etapas

### 1. MLP

- Implementar e executar o treinamento de uma MLP **sem camadas convolucionais**.
- Apresentar gráficos de perda e acurácia por época para treino e validação.
- Explicar como as duas métricas são calculadas e por que elas não são necessariamente proporcionais.
- Analisar se o desempenho é satisfatório, considerando treino, validação e possível generalização.

### 2. CNN com aumento de dados

- Implementar e executar o treinamento de uma CNN com aumento de dados, seguindo a aula de referência.
- Explicar a finalidade do aumento de dados e sua contribuição para a generalização.
- Apresentar gráficos de perda e acurácia por época para treino e validação.
- Analisar o desempenho e os indícios de sobreajuste ou subajuste, comparando os resultados com os da MLP.

### 3. Experimentação e otimização de hiperparâmetros

- Definir e informar os valores ou intervalos dos hiperparâmetros testados na CNN.
- Executar testes com diferentes combinações e indicar os melhores valores encontrados.
- Apresentar uma tabela comparativa com as combinações testadas e os respectivos resultados.
- Analisar quais parâmetros e combinações mais interferiram no desempenho e explicar os possíveis motivos, conforme os resultados observados.

### 4. Avaliação final

- Utilizar o melhor modelo encontrado na otimização para avaliar o conjunto de teste.
- Apresentar perda e acurácia finais, matriz de confusão e discussão sobre generalização em dados não vistos.

## Cuidados com os experimentos e as análises

- Utilizar o Fashion-MNIST fornecido e aproveitar a preparação de dados existente quando estiver correta; não substituir o dataset pelo MNIST de dígitos usado nas aulas.
- Manter os conjuntos de treino, validação e teste separados. Usar a validação para comparar e selecionar modelos; reservar o teste para a avaliação final do melhor modelo.
- Aplicar aumento de dados somente durante o treinamento, preservando validação e teste para avaliação sem transformações aleatórias.
- Manter a mesma divisão de dados para comparar os experimentos e usar o controle simples de aleatoriedade das referências, sem prometer reprodução exata apenas por fixar sementes.
- Basear tabelas, gráficos e conclusões em execuções reais. Não inventar métricas, resultados, saídas, nomes de participantes ou contribuições.
- Não afirmar que uma CNN superou a MLP ou que houve melhoria com a otimização sem evidências. Reconhecer as limitações das comparações e não atribuir causalidade isolada a um hiperparâmetro quando vários mudaram juntos.

## Entrega e verificação

- Concentrar a solução no notebook Jupyter da equipe, com código, experimentos, resultados e **saídas salvas**.
- Incluir os nomes dos estudantes e uma frase descrevendo a contribuição de cada participante; solicitar essas informações se estiverem ausentes quando necessárias.
- Inserir explicações conceituais em células de texto **antes de cada implementação em código**.
- Garantir que o notebook contemple carregamento dos dados, MLP, CNN, otimização e teste final, com os gráficos e a matriz de confusão exigidos.
- Executar as células em ordem e verificar a coerência entre código, saídas e análises. Se uma etapa não tiver sido executada, informar isso claramente, sem apresentá-la como concluída.
- Preservar as saídas reais no arquivo entregue e respeitar os demais requisitos de entrega, referências, autoria e registro de uso de IA previstos no enunciado.
- Antes de finalizar uma alteração, conferir se ela atende a um requisito solicitado e segue o código de referência. Remover acréscimos desnecessários.
