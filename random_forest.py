"""
 Para simplificar, usaremos o já familiar conjunto de dados Iris nas seções a seguir. Convenientemente, o 
conjunto de dados Iris já está disponível através do scikit-learn, uma vez que é um conjunto de dados simples, 
mas popular, que é freqüentemente usado para testar e experimentar algoritmos. Além disso,
 usaremos apenas dois recursos do conjunto de dados de flor da Iris para fins de visualização.

 nós treinamos uma floresta aleatória a partir de 10 árvores de decisão através do parâmetro n_estimators
 e usamos o critério de entropia como uma medida de impureza para dividir os nós. Embora estejamos
 desenvolvendo uma floresta aleatória muito pequena a partir de um conjunto de dados de treinamento
 muito pequeno, usamos o parâmetro n_jobs para fins de demonstração, o que nos permite paralelizar
 o treinamento do modelo usando vários núcleos do nosso computador (aqui, dois)
"""

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn import datasets
import numpy as np
from sklearn.preprocessing import StandardScaler

"""
Atribuiremos o comprimento da pétala e a largura da pétala das 150 amostras de flores à matriz 
característica X e os rótulos de classe correspondentes das espécies de flores ao vetor y.

Se executamos np.unique (y) para retornar os diferentes rótulos de classe armazenados na íris. alvo, 
veríamos que os nomes das classes de flores Iris, Iris-Setosa, Iris-Versicolor e Iris-Virginica, já 
são armazenados como inteiros (0, 1, 2), o que é recomendado para o desempenho ideal de muitas 
bibliotecas de aprendizado de máquina .

"""
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()
