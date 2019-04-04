"""
Colocando atributos na mesma escala
"""

"""
O dimensionamento de recursos é uma etapa crucial em nosso pipeline de 
pré-processamento que pode ser facilmente esquecido. Árvores de decisão e 
florestas aleatórias são um dos poucos algoritmos de aprendizado de máquina em 
que não precisamos nos preocupar com o dimensionamento de recursos. No entanto, 
a maioria dos algoritmos de aprendizado de máquina e otimização se comportam 
muito melhor se os recursos estiverem na mesma escala,
"""

"""
Agora, existem duas abordagens comuns para trazer recursos diferentes para a 
mesma escala: normalização e padronização. Esses termos são freqüentemente 
usados de maneira muito frouxa em campos diferentes, e o significado deve 
ser derivado do contexto. Na maioria das vezes, a normalização refere-se 
ao reescalonamento dos recursos para um intervalo de [0, 1], que é um 
caso especial de escala min-max. Para normalizar nossos dados, podemos 
simplesmente aplicar o escalonamento (i) min-max a cada coluna de recurso, 
onde o novo valor x norma de uma amostra x (i) pode ser calculado da seguinte forma:

x(i,normal) = (x(i) - x(min)) / (x(max) - x(min))

Aqui, x é uma amostra particular, x min é o menor valor em uma coluna de recurso e x max 
o maior valor, respectivamente.
O procedimento de escalonamento min-max é implementado no scikit-learn e pode 
ser usado da seguinte maneira:
"""


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)


"""
Embora a normalização por meio do escalonamento min-max seja uma técnica comumente 
usada quando precisamos de valores em um intervalo limitado, a padronização pode ser 
mais prática para muitos algoritmos de aprendizado de máquina. A razão é que muitos 
modelos lineares, como a regressão logística e o SVM que lembramos do Capítulo 3, 
inicializam os pesos em 0 ou pequenos valores aleatórios próximos a 0. Usando a 
padronização, Centralize as colunas de recurso na média 0 com o desvio padrão 1, 
de modo que as colunas de feições tenham a forma de uma distribuição normal, o que 
facilita o aprendizado dos pesos. Além disso, a padronização mantém informações 
úteis sobre valores discrepantes e torna o algoritmo menos sensível a eles, em 
contraste com a escala min-max, que dimensiona os dados para um intervalo 
limitado de valores.
O procedimento de padronização pode ser expresso pela seguinte equação:

x(i, std) = (x(i) - μ(x)) / σ(x)

Aqui, μ(x) é a média da amostra de uma coluna característica particular e σ(x) o desvio padrão correspondente, respectivamente.
Semelhante ao MinMaxScaler, o scikit-learn também implementa uma classe para padronização:
"""

from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)



"""
Novamente, também é importante destacar que ajustamos o StandardScaler apenas uma vez 
nos dados de treinamento e usamos esses parâmetros para transformar o conjunto de 
testes ou qualquer novo ponto de dados.
"""
