from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import utility_random_forests as urf

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


"""
@param
	n_estimators = default 10. O número de árvores que você deseja construir dentro de uma Floresta Aleatória 
					antes de agregar as previsões
	criterion    = Mede a qualidade de cada divisão
	max_features = default = auto. O número máximo de recursos considerados ao encontrar a melhor divisão
	max_depth    = default = none. Isso seleciona o quão profundo você quer fazer suas árvores
	min_samples_split = default = 2. O número mínimo de amostras que devem estar presentes de seus dados para 
						que uma divisão ocorra
	min_samples_leaf  = default = 1. Determinar o tamanho mínimo do nó final de cada árvore de decisão
	min_weight_fraction_leaf = default = 0. bastante semelhante a min_samples_leaf, mas usa uma fração do total 
							   de observações em vez disso
	max_leaf_nodes    = default = none.  aumenta a árvore da melhor forma, resultando em uma redução relativa na impureza
	min_impurity_decrease = default 0. Um nó será dividido se essa divisão induzir uma diminuição da impureza maior 
							ou igual a esse valor
	bootstrap    = default true. Se você deve ou não fazer o bootstrap de suas amostras ao construir árvores
	oob_score    = default false. Este é um método de validação cruzada que é muito semelhante a uma técnica de 
					validação leave-one-out, na qual o desempenho estimado generalizado de um modelo é treinado 
					em amostras n-1 dos dados. No entanto, oob_score é muito mais rápido porque captura todas as 
					observações usadas nas árvores e descobre a pontuação máxima para cada base de observação 
					nas árvores que não usaram essa observação para treinar
	n_jobs       = default 1. Isso permite que o computador saiba quantos processadores ele pode usar. O valor 
					padrão de 1 significa que ele pode usar apenas um processador. Se você usar -1, significa 
					que não há restrição de quanto poder de processamento o código pode usar. Definir seus 
					n_jobs como -1 geralmente levará a um processamento mais rápido.
	random_state = default = None. Como o bootstrapping gera amostras aleatórias, muitas vezes é difícil duplicar 
					exatamente os resultados. Esse parâmetro torna fácil para os outros replicarem seus resultados 
					se receberem os mesmos dados e parâmetros de treinamento.
	verbose      = default = 0. configurando a saída de registro que lhe fornece atualizações constantes sobre 
					o que o modelo está fazendo enquanto é processado.
	warm_start   = defult = False. No False, ele se ajusta a uma nova floresta a cada vez, em vez de quando é 
					True, adiciona estimadores e reutiliza a solução do ajuste anterior. É usado principalmente 
					quando você está usando a seleção de recursos recursivos. Isso significa que, quando você 
					soltar alguns recursos, outros recursos ganharão importância e para “apreciar” as árvores, 
					eles devem ser reutilizados. É freqüentemente usado com eliminação reversa em modelos de 
					regressão e não é usado com freqüência em modelos de classificação
	class_weight  = default: None. também conhecido como “subamostra balanceada”, pondera as classes. Se você 
					não colocar algo aqui, assumirá que todas as classes têm um peso de 1, mas se você tiver 
					um problema com várias saídas, uma lista de dicionários será usada como as colunas de y. 
					Quando o modo “balanceado” é usado, os valores y ajustam automaticamente seus “pesos 
					inversamente proporcionais às freqüências de classe” nos dados usando 
					“n_samples / (n_classes * np.bincount(y))”
					
"""
    forest = RandomForestClassifier(n_estimators= urf.get_n_estimators(6, 5), 
								criterion='entropy',
								max_features = 'auto',
								max_depth = urf.get_max_depth(6),
								min_samples_split = 2,
								min_samples_leaf = 1,
								min_weight_fraction_leaf = 0,
								max_leaf_nodes = None,
								min_impurity_decrease = 0,
								bootstrap = True,
								oob_score = True,
								n_jobs = -1,
								random_state = 1,
								verbose = 1,
								warm_start = False,
								class_weight = None
								)
forest = forest.fit(X_train, y_train)

"""
 @Attrib
"""

# A coleção de sub-estimadores ajustados.
## forest.estimators_
print(len(forest.estimators_))

# As classes de saída
## forest.classes_
print('Classes = ' + str(forest.classes_))

# O número de classes de saída
## forest.n_classes_
print('N Classes = ' + str(forest.n_classes_))

# O número de recursos quando o fit é executado.
## forest.n_features_
print('N features = ' + str(forest.n_features_))

# O número de saídas quando o fit é realizado.
## forest.n_outputs_ 
print('N outputs = ' + str(forest.n_outputs_ ))

# Retornar uma matriz das importâncias do recurso (quanto mais alto, mais importante o recurso).
## forest.feature_importances_
print("Feature Importances")
print(forest.feature_importances_)

# Pontuação do conjunto de dados de treinamento obtido usando uma estimativa out-of-bag
## forest.oob_score_
print("Oob Score = " + str(forest.oob_score_))

# Função de decisão computada com estimativa fora do saco no conjunto de treinamento. 
# Se n_estimators for pequeno, pode ser possível que um ponto de dados nunca tenha sido 
# deixado de fora durante o bootstrap. Nesse caso, oob_decision_function_ pode conter NaN
## forest.oob_decision_function_ 
print('Oob Decision Function')
print(forest.oob_decision_function_ )

"""
 @Method
"""

# Retorna a precisão média nos dados e rótulos de teste fornecidos.
# Na classificação de vários rótulos, essa é a precisão do subconjunto, que é uma métrica rígida, pois 
# é necessário para cada amostra que cada conjunto de rótulos seja corretamente previsto. 
fscore = forest.score(X_train, y_train)
print('Score')
print(fscore)

forest.estimators_ = forest.estimators_.pop()
print(len(forest.estimators_))
