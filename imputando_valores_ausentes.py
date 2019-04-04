"""
Imputando valores ausentes
"""

"""
Muitas vezes, a remoção de exemplos ou apagar colunas de atributos inteiras
simplesmente não é viável, porque podemos perder muitos dados valiosos. Nesse
caso, podemos usar diferentes técnicas de interpolação para estimar os valores
ausentes das outras amostras de treinamento em nosso conjunto de dados.
Uma das técnicas de interpolação mais comuns é a imputação média, em que
simplesmente substituímos o valor ausente pelo valor médio de toda a coluna do
recurso. Uma maneira conveniente de conseguir isso é usando a classe Imputer
do scikit-learn, conforme mostrado no código a seguir:

Aqui, substituímos cada valor NaN pela média correspondente, que é calculada
separadamente para cada coluna de recurso. Se mudássemos o eixo de ajuste = 0
[para o eixo = 1, calcularíamos o meio da linha. Outras opções para o
parâmetro strategy são median ou most_frequent, em que o último substitui os
valores omissos pelos valores mais frequentes. Isso é útil para imputar
valores de recursos categóricos.

A classe Imputer pertence às chamadas classes de transformadores em
scikit-learn que são usadas para transformação de dados.

Os dois métodos essenciais desses estimadores são fit e transform. O método 
de ajuste (fit) é usado para aprender os parâmetros dos dados de treinamento e o método 
de transformação (transform) usa esses parâmetros para transformar os dados.
Qualquer array de dados a ser transformado precisa tem o mesmo número de recursos 
que a matriz de dados usada para ajustar o modelo.

Os classificadores que usamos no Capítulo 3, Um Tour de Classificadores de 
Aprendizado de Máquina Usando o Scikit-Learn, pertencem aos chamados estimadores em 
scikit-learn com uma API que é conceitualmente muito semelhante à classe de transformadores. 
Os estimadores têm um método de previsão, mas também podem ter um método de transformação, 
como veremos mais adiante. Como você deve se lembrar, também usamos o método de ajuste 
para aprender os parâmetros de um modelo quando treinamos esses estimadores para classificação. 
No entanto, em tarefas de aprendizado supervisionadas, fornecemos adicionalmente os rótulos 
de classe para ajustar o modelo, que podem ser usados para fazer previsões sobre novas 
amostras de dados por meio do método de previsão,
"""

import pandas as pd
from io import StringIO
from sklearn.preprocessing import Imputer


csv_data = '''A,B,C,D
... 1.0,2.0,3.0,4.0
... 5.0,6.0,,8.0
... 0.0,11.0,12.0,'''
# If you are using Python 2.7, you need
# to convert the string to unicode:
# csv_data = unicode(csv_data)
df = pd.read_csv(StringIO(csv_data))

print(df)

imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)

print(imputed_data)
