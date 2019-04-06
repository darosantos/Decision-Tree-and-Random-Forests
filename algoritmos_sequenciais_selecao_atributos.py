"""
Algoritmos sequenciais de seleção de atributos
"""

"""
Uma maneira alternativa de reduzir a complexidade do modelo e evitar 
overfitting é a "redução da dimensionalidade" por meio da seleção de 
recursos, o que é especialmente útil para modelos não regularizados.
Existem duas categorias principais de técnicas de redução de 
dimensionalidade: seleção de recursos e extração de características. 
Usando a seleção de recursos, selecionamos um subconjunto dos recursos 
originais. Na extração de características, derivamos informações do 
conjunto de recursos para construir um novo subespaço de recursos. 
"""

"""
Algoritmos de seleção de atributos sequenciais são uma família de 
algoritmos de busca gulosos que são usados para reduzir um espaço 
de características d-dimensional inicial para um subespaço de 
características k-dimensionais onde k <d. A motivação por trás dos 
algoritmos de seleção de recursos é selecionar automaticamente um 
subconjunto de recursos que são mais relevantes para o problema 
para melhorar a eficiência computacional ou reduzir o erro de 
generalização do modelo removendo recursos irrelevantes ou ruído, 
o que pode ser útil para algoritmos que não apoiar a regularização. 
Um algoritmo clássico de seleção de características seqüenciais é o 
Sequential Backward Selection (SBS), que visa reduzir a dimensionalidade 
do subespaço da característica inicial com um decaimento mínimo no 
desempenho do classificador para melhorar a eficiência computacional. 
Em certos casos, o SBS pode até melhorar o poder preditivo do modelo 
se um modelo sofrer um overfitting.
"""

"""
Algoritmos gananciosos fazem escolhas ótimas localmente em cada 
estágio de um problema de busca combinatorial e geralmente geram 
uma solução sub-ótima para o problema em contraste com algoritmos 
de busca exaustivos, que avaliam todas as combinações possíveis 
e garantem encontrar a solução ótima. No entanto, na prática, 
uma busca exaustiva é frequentemente inviável em termos computacionais, 
enquanto os algoritmos gulosos permitem uma solução computacionalmente 
mais eficiente, menos complexa.
"""

"""
A idéia por trás do algoritmo SBS é bastante simples: o SBS remove 
sequencialmente os recursos do subconjunto de recursos completo até 
que o novo subespaço de recursos contenha o número desejado de 
recursos. Para determinar qual recurso deve ser removido em cada 
estágio, precisamos definir a função de critério J que queremos 
minimizar. O critério calculado pela função de critério pode ser 
simplesmente a diferença no desempenho do classificador após e antes 
da remoção de um recurso específico. Em seguida, o recurso a ser 
removido em cada estágio pode ser simplesmente definido como o recurso 
que maximiza esse critério; ou, em termos mais intuitivos, em cada 
estágio, eliminamos o recurso que causa menos perda de desempenho 
após a remoção.
"""

"""
Infelizmente, o algoritmo SBS ainda não está implementado no 
scikit-learn. Mas como é tão simples, vamos implementá-lo em 
Python a partir do zero:
"""

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


class SBS():
	def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
		self.scoring = scoring
		self.estimator = clone(estimator)
		self.k_features = k_features
		self.test_size = test_size
		self.random_state = random_state
		
	def fit(self, X, y):
		X_train, X_test, y_train, y_test = \
		train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
		dim = X_train.shape[1]
		self.indices_ = tuple(range(dim))
		self.subsets_ = [self.indices_]
		score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
		self.scores_ = [score]
		while dim > self.k_features:
			scores = []
			subsets = []
			for p in combinations(self.indices_, r=dim-1):
				score = self._calc_score(X_train, y_train, X_test, y_test, p)
				scores.append(score)
				subsets.append(p)
			best = np.argmax(scores)
			self.indices_ = subsets[best]
			self.subsets_.append(self.indices_)
			dim -= 1
			self.scores_.append(scores[best])
		self.k_score_ = self.scores_[-1]
		return self
		
	def transform(self, X):
		return X[:, self.indices_]
		
	def _calc_score(self, X_train, y_train, X_test, y_test, indices):
		self.estimator.fit(X_train[:, indices], y_train)
		y_pred = self.estimator.predict(X_test[:, indices])
		score = self.scoring(y_test, y_pred)
		return score
		

"""
Na implementação anterior, definimos o parâmetro k_features 
para especificar o número desejado de recursos que queremos retornar. 
Por padrão, usamos a accuracy_score do scikit-learn para avaliar o 
desempenho de um modelo e um estimador para classificação nos subconjuntos
 de recursos. Dentro do loop while do método fit, os subconjuntos 
 de recursos criados pela função itertools.combination são avaliados 
 e reduzidos até que o subconjunto de recursos tenha a dimensionalidade 
 desejada. Em cada iteração, a pontuação de precisão do melhor 
 subconjunto é coletada em uma lista self.scores_ com base no conjunto 
 de dados de teste internamente criado X_test. Usaremos essas pontuações 
 mais tarde para avaliar os resultados. Os índices da coluna do 
 subconjunto final de recursos são atribuídos a self.indices_, que 
 podemos usar por meio do método transform para retornar uma nova 
 matriz de dados com as colunas de recursos selecionadas. Observe que, 
 em vez de calcular o critério explicitamente dentro do método de 
 ajuste, simplesmente removemos o recurso que não está contido no 
 subconjunto de recursos de melhor desempenho.
"""

"""
Agora, vamos ver nossa implementação do SBS em ação usando o 
classificador KNN do scikit-learn:
"""

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)


"""
Embora nossa implementação do SBS já divida o conjunto de dados 
em um conjunto de dados de teste e treinamento dentro da função de 
ajuste, ainda alimentamos o conjunto de dados de treinamento X_train 
para o algoritmo. O método de ajuste SBS criará novos subconjuntos de 
treinamento para teste (validação) e treinamento, e é por isso que esse
 conjunto de testes também é chamado de conjunto de dados de validação. 
 Essa abordagem é necessária para impedir que nosso conjunto de testes 
 original se torne parte dos dados de treinamento.
"""

"""
Lembre-se de que nosso algoritmo SBS coleta as pontuações do melhor 
subconjunto de recursos em cada estágio, portanto, vamos passar para 
a parte mais interessante de nossa implementação e plotar a precisão 
da classificação do classificador KNN que foi calculado no conjunto 
de dados de validação. O código é o seguinte:
"""

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()


"""
Para satisfazer nossa própria curiosidade, vamos ver quais são esses 
cinco recursos que renderam um desempenho tão bom no conjunto de 
dados de validação:
"""

k5 = list(sbs.subsets_[8])
print(df_wine.columns[1:][k5])
##Index(['Alcohol', 'Malic acid', 'Alcalinity of ash', 'Hue', 'Proline'], dtype='object')


"""
Em seguida, vamos avaliar o desempenho do classificador KNN no conjunto 
de testes original:
"""

knn.fit(X_train_std, y_train)
print('Training accuracy:', knn.score(X_train_std, y_train))
## Training accuracy: 0.983870967742
print('Test accuracy:', knn.score(X_test_std, y_test))
## Test accuracy: 0.944444444444

"""
No código anterior, usamos o conjunto de recursos completo e obtivemos 
uma precisão de ~ 98,4% no conjunto de dados de treinamento. No entanto, 
a precisão no conjunto de dados de teste foi ligeiramente menor 
(~ 94,4 por cento), o que é um indicador de um leve grau de overfitting. 
Agora vamos usar o subconjunto de 5 recursos selecionado e ver o desempenho do KNN:
"""

knn.fit(X_train_std[:, k5], y_train)
print('Training accuracy:', knn.score(X_train_std[:, k5], y_train))
##Training accuracy: 0.959677419355
print('Test accuracy:', knn.score(X_test_std[:, k5], y_test))
##Test accuracy: 0.962962962963

"""
Existem muitos algoritmos de seleção de recursos disponíveis via 
scikit-learn. Esses incluem eliminação recursiva para trás com base 
em pesos de recursos, métodos baseados em árvore para selecionar 
recursos por importância e testes estatísticos univariados.
"""