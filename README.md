# Trabalho 02 — Redes Neurais

UFSC · Sistemas Inteligentes (INE5633) · Profa. Nathalia da Cruz Alves.

Classificação de imagens do Fashion-MNIST com MLP e CNN, comparação das curvas
de aprendizagem e otimização de hiperparâmetros. **Entrega: 15/11/2026**,
em grupo de 2 a 3 estudantes, conforme o enunciado local.

## Começar

Com o Docker Desktop aberto, execute na raiz deste repositório:

```bash
docker compose up -d --build
```

Na primeira execução, aguarde o download da imagem e a instalação das dependências.
Abra <http://localhost:8888/lab> e abra `Trabalho_02_Redes_Neurais.ipynb`,
que aparece diretamente na pasta inicial. Para executar o trabalho, rode as células
de cima para baixo.
O notebook inicial prepara o ambiente, verifica/baixa os dados, mostra exemplos e
cria a divisão treino/validação. As etapas de modelos e análises estão indicadas
para desenvolvimento pela equipe; nenhum treinamento foi realizado.

### Acompanhar as alterações no notebook

A pasta do repositório é compartilhada diretamente com o contêiner, sem cópias.
O Jupyter Collaboration verifica alterações no disco a cada segundo e atualiza
o notebook aberto quando uma nova versão é salva externamente. Esse comportamento
segue a [configuração de alterações externas do Jupyter Collaboration](https://jupyterlab-realtime-collaboration.readthedocs.io/en/latest/configuration.html#files-changed-on-disk).

Deixe o notebook aberto ao lado do chat para acompanhar as alterações salvas.
Enquanto ele estiver sendo editado externamente, use a tela apenas para acompanhar:
alterações externas substituem o conteúdo da sessão aberta. Combine uma pausa nas
edições externas antes de editar ou executar células pela interface.
Salvar código não executa as células; os resultados aparecem quando a execução
é realizada e suas saídas são salvas ou produzidas na interface do Jupyter.

### Gerenciar o ambiente

Para acompanhar a inicialização ou investigar um problema:

```bash
docker compose logs -f jupyter
```

Para parar, preservando os arquivos do trabalho:

```bash
docker compose down
```

Nas próximas vezes, use `docker compose up -d`. Se mudar `requirements.txt` ou
o `Dockerfile`, execute novamente `docker compose up -d --build`.
O serviço fica acessível somente neste computador, sem token ou senha, como no
ambiente do Trabalho 1. Se a porta 8888 estiver ocupada pelo Jupyter do Trabalho 1,
pare aquele ambiente antes de iniciar este.

### Alternativa sem Docker

Para usar um ambiente Python local, execute a partir desta pasta:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/baixar_dados.py
python -m jupyter lab --YDocExtension.file_poll_interval=1
```

Os quatro arquivos do dataset ficam em `dados/fashion-mnist/` e são conferidos por
MD5. A pasta de dados, os modelos e os resultados externos são ignorados pelo Git.
**Mantenha as saídas das células no notebook para a entrega.**
Keras Tuner e TensorBoard constam do mesmo `requirements.txt` usado no Docker e localmente.
Para atualizar a imagem existente, reconstrua com o comando Docker acima.
A aula 23 também permite instalar no kernel atual com `%pip install keras-tuner tensorboard`.

## Organização

```text
Trabalho_02_Redes_Neurais.ipynb       Notebook da equipe
Trabalho Prático 2 - Redes Neurais.pdf  Enunciado original
Aulas_Praticas/                      Notebooks de referência da professora
Conteudo/                           Slides das aulas 17 a 20 e 22
dados/                              Origem, licença e dataset local
scripts/baixar_dados.py              Download e verificação reproduzíveis
modelos/                            Modelos gerados (ignorados)
resultados/                         Tabelas e figuras geradas (ignoradas)
Dockerfile                         Imagem Jupyter com as dependências do trabalho
docker-compose.yml                 Inicialização e compartilhamento do repositório
requirements.txt                    Dependências para Docker e ambiente local
```

Os materiais fornecidos foram preservados com os nomes e a autoria originais.
Esta pasta tem um repositório Git independente, com branch principal `main`.
Os comandos Git executados aqui operam apenas sobre o trabalho.
Para publicar no GitHub, configure o remoto deste repositório com a URL SSH
do repositório de destino e envie a branch `main`.

## Requisitos e andamento

Fonte: [Trabalho Prático 2 — Redes Neurais](<Trabalho Prático 2 - Redes Neurais.pdf>).

- [x] Estrutura inicial, dependências e `.gitignore`.
- [x] Download reproduzível e preparação inicial do Fashion-MNIST.
- [ ] Informar os nomes dos participantes e uma frase sobre a contribuição de cada um.
- [ ] MLP sem convoluções: treinar, plotar loss/acurácia de treino e validação,
      explicar o cálculo das métricas e por que não são proporcionais.
- [ ] CNN com aumento de dados: treinar, justificar as transformações, plotar
      curvas e comparar com a MLP, discutindo sobreajuste/subajuste.
- [ ] Testar combinações de hiperparâmetros da CNN; apresentar intervalos,
      tabela comparativa, melhores valores e análise das diferenças.
- [ ] Avaliar a melhor CNN no teste: loss, acurácia, matriz de confusão e discussão
      da generalização. Selecionar hiperparâmetros apenas pela validação.
- [ ] Explicar os conceitos antes de cada implementação e salvar as saídas reais.
- [ ] Registrar todas as referências e o uso de IA com sua finalidade.
- [ ] Gravar vídeo de até 8 minutos e disponibilizar os links de entrega.

O enunciado sugere distribuir o vídeo em: introdução (1 min), MLP (2 min),
CNN (2 min), otimização (1 min) e resultados/conclusões (2 min).

## Referências e uso de IA

- Slides e aulas práticas da Profa. Nathalia da Cruz Alves, incluídos nesta pasta.
- [Fashion-MNIST — fonte oficial, classes, checksums e licença](https://github.com/zalandoresearch/fashion-mnist).
- Xiao, Rasul e Vollgraf (2017), [artigo do Fashion-MNIST](https://arxiv.org/abs/1708.07747).
- OpenAI Codex: utilizado na organização inicial do repositório e na criação do
  notebook base, do script de download e da documentação. Atualizar este registro
  se houver outros usos. O enunciado recomenda IA apenas como apoio à pesquisa;
  a equipe deve revisar a base gerada e desenvolver os experimentos e análises,
  observando os critérios de autoria e uso de IA descritos no PDF.
