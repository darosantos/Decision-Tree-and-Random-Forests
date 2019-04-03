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

# ver esses pacotes majorityvoting, rotation-forest, fasttrees, pandas-transformers

# [0]
import numpy as np
import matplotlib.pyplot as plt
# [1]
from sklearn import datasets
# [2]
## O pacote abaixo caiu em desuso e a segunda opção é a sugerida
## https://scikit-learn.org/stable/modules/cross_validation.html
## conda update scikit-learn
## pip install -U scikit-learn
#### from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
# [3]
from sklearn.preprocessing import StandardScaler
# [4]
from sklearn.ensemble import RandomForestClassifier
# [5]
from sklearn.metrics import accuracy_score
# [6]
from matplotlib.colors import ListedColormap



def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
	# setup marker generator and color map
	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])
	# plot the decision surface
	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
	np.arange(x2_min, x2_max, resolution))
	Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
	Z = Z.reshape(xx1.shape)
	plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())
	# plot all samples
	X_test, y_test = X[test_idx, :], y[test_idx]
	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
		alpha=0.8, c=cmap(idx),
		marker=markers[idx], label=cl)
	# highlight test samples
	if test_idx:
		X_test, y_test = X[test_idx, :], y[test_idx]
		plt.scatter(X_test[:, 0], X_test[:, 1], c='', alpha=1.0, linewidth=1, marker='o', s=55, label='test set')


"""
Atribuiremos o comprimento da pétala e a largura da pétala das 150 amostras de flores à matriz 
característica X e os rótulos de classe correspondentes das espécies de flores ao vetor y.

Se executamos np.unique (y) para retornar os diferentes rótulos de classe armazenados na íris. alvo, 
veríamos que os nomes das classes de flores Iris, Iris-Setosa, Iris-Versicolor e Iris-Virginica, já 
são armazenados como inteiros (0, 1, 2), o que é recomendado para o desempenho ideal de muitas 
bibliotecas de aprendizado de máquina .

"""
# [1]
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

"""
Para avaliar o desempenho de um modelo treinado em dados não vistos, dividiremos o conjunto 
de dados em conjuntos de dados de treinamento e teste separados.

Usando a função train_test_split da cross_validation do scikit-learn
módulo, dividimos aleatoriamente as matrizes X e Y em 30% de dados de teste (45 amostras) 
e 70% de dados de treinamento (105 amostras).

Usamos o parâmetro random_state para a reprodutibilidade do embaralhamento inicial do 
conjunto de dados de treinamento após cada época.
"""
# [2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

"""
Muitos algoritmos de aprendizado de máquina e otimização também exigem o dimensionamento de recursos 
para um ótimo desempenho, conforme nos lembramos do exemplo de gradiente descendente no 
Capítulo 2, Algoritmos de Aprendizado de Máquina de Treinamento para Classificação. Aqui, 
padronizaremos os recursos usando a classe StandardScaler do módulo de pré-processamento do scikit-learn.

Usando o código anterior, carregamos a classe StandardScaler do módulo de pré-processamento e inicializamos 
um novo objeto StandardScaler que atribuímos à variável sc. Usando o método de ajuste, o StandardScaler 
estimou os parâmetros μ (média da amostra) e σ (desvio padrão) para cada dimensão do recurso a partir 
dos dados de treinamento. Ao chamar o método transform, padronizamos os dados de treinamento usando os 
parâmetros estimados μ e σ. Observe que usamos os mesmos parâmetros de escala para padronizar o conjunto 
de testes, de modo que ambos os valores no conjunto de dados de treinamento e teste sejam comparáveis entre si.
"""
# [3]
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

"""
Tendo padronizado os dados de treinamento, agora podemos treinar um modelo. A maioria dos 
algoritmos no scikit-learn já suporta a classificação multiclasse por padrão através do método 
One-vs.-Rest (OvR), que nos permite alimentar as três classes de flores ao perceptron de uma só vez.
"""
#[4]
forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)

"""
O Scikit-learn também implementa uma grande variedade de diferentes métricas de desempenho que estão 
disponíveis através do módulo de métricas. Por exemplo, podemos calcular a precisão da classificação 
do perceptron no conjunto de testes da seguinte forma:

Aqui, y_test são os verdadeiros rótulos de classe e y_pred são os rótulos de classe que previmos 
anteriormente.
"""
#[5]
##y_pred = classifier.predict(X_test_std)
##print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

"""
Com a pequena modificação que fizemos na função plot_decision_regions (destacada no código anterior), 
podemos agora especificar os índices das amostras que queremos marcar nos gráficos resultantes. 
"""
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=forest, test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()


