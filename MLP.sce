//Aluno: Francilândio Lima Serafim (472644)

clc;
clear all;

base = csvRead('dermatology.data'); //Lendo o arquivo da base de dados

base(:, [1,35]) = base(:, [35,1]); //Colocando a coluna das classes no início

base = gsort(base, 'lr', 'i'); //Organizando as linhas de acordo com as classes

base(:, [1, 34]) = base(:, [34, 1]); //Trazendo a coluna com dados nan para o início, facilitando a remoção das linhas

index = find(isnan(base)); //Pegando os números das linhas que possuem nan

n = length(index); //Número de elementos nan

for i = 1:n
    base(index(i)-(i-1), :) = []; //Remoção das linhas com nan
end

base(:, [1, 34]) = base(:, [34, 1]); //Desfazendo a última permutação de colunas

//Enumerar quantos elementos cada classe possui
number_el = zeros(1, 6); //vetor para armazenar elementos por classe
for i = 1:size(base)(1)
    select base(i, 1)
    case 1
        number_el(1) = number_el(1) + 1;
    case 2
        number_el(2) = number_el(2) + 1;
    case 3
        number_el(3) = number_el(3) + 1;
    case 4
        number_el(4) = number_el(4) + 1;
    case 5
        number_el(5) = number_el(5) + 1;
    case 6
        number_el(6) = number_el(6) + 1;
    end
end
//Classes: 1-> 1:111(111) - 2-> 112:171(60) - 3-> 172:242(71) - 4-> 243:290(48) - 5-> 291:338(48) - 6-> 339:358(20)

/*=============Separação de dados para treino e teste:(50% cada)==============
- Treino:                                   - Teste:
Classe 1-> 1:56 (56)                        Classe 1-> 57:111 (55)
Classe 2-> 112:141 (30)                     Classe 2-> 142:171 (30)
Classe 3-> 172:207 (36)                     Classe 3-> 208:242 (35)
Classe 4-> 243:266 (24)                     Classe 4-> 267:290 (24)
Classe 5-> 291:314 (24)                     Classe 5-> 315:338 (24)
Classe 6-> 339:348 (10)                     Classe 6-> 349:358 (10)
*/

ind_treino = [1:56, 112:141, 172:207, 243:266, 291:314, 339:348]; //Intervalos dos dados de treino
ind_teste = [57:111, 142:171, 208:242, 267:290, 315:338, 349:358]; //Intervalos dos dados de teste

P = base(:, 2:35)'; //Matriz com atributos
T = base(:, 1)'; //Vetor de rótulos

//Codificação das classes
descritor = zeros(6, 358); //Matriz para codificar as classes - Ex: classe 1 = [1 0 0 0 0 0]; classe 4 = [0 0 0 1 0 0]; classe 6 = [0 0 0 0 0 1]
for i = 1:size(base)(1)
    select T(i)
    case 1
        descritor(:, i) = [1 0 0 0 0 0]';
    case 2
        descritor(:, i) = [0 1 0 0 0 0]';
    case 3
        descritor(:, i) = [0 0 1 0 0 0]';
    case 4
        descritor(:, i) = [0 0 0 1 0 0]';
    case 5
        descritor(:, i) = [0 0 0 0 1 0]';
    case 6
        descritor(:, i) = [0 0 0 0 0 1]';
    end
end

//Normalizando atributos pelo método z-score
for i = 1:34 
    P(i, :) = (P(i, :) - mean(P(i, :)))/stdev(P(i, :));
end

P_treino = P(:, ind_treino); //Atributos de treino
T_treino = descritor(:, ind_treino); //Rótulos dos atributos de treino

P_teste = P(:, ind_teste); //Atributos de teste
T_teste = descritor(:, ind_teste); //Rótulos dos atributos de teste

//34 atributos; 155 neurônios escolhidos ao acaso; 6 rótulos
W = ann_FFBP_gd(P_treino, T_treino, [34 155 6]); //Treinando o modelo MLP 

C = ann_FFBP_run(P_teste, W); //Testando

//Fazendo a validação do modelo com os dados de teste
cont = 0;
for i = 1:size(T_teste)(2)
    [a b] = max(T_teste(:, i)); //a = maior valor, b = índice
    [c d] = max(C(:, i));
    if b == d
        cont = cont + 1;
    end
end

//Resultado
cont = 100 * (cont/size(T_teste)(2)); //Porcentagem de acertos nos dados de teste.
printf('\n\n O percentual de acertos dessa rede MLP é %3.2f%%.', cont);

