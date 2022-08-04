# Inteligencia-Computacional-2
Trabalho desenvolvido para a disciplina de Inteligência Computacional.

Foi implementada uma rede neural RBF e utilizada uma rede neural MLP (por meio do Neural
Network Module, versão 3.0, do Scilab) para classificar a base de dados disponível em
(https://archive.ics.uci.edu/ml/datasets/Dermatology). Nessa base as amostras com
dados faltantes foram removidas e os atributos normalizados por z-score.
Como estratégia de validação, foram utilizadas 50% das amostras de cada classe para treinamento
da rede neural e o restante para teste. O resultado mostrado foi a percentagem de acertos nas
amostras de teste.

O código foi desenvolvido em SCILAB 6.1.0.
