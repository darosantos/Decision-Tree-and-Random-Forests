"""
Particionando um conjunto de dados em conjuntos de treinamento e teste
"""

"""
O conjunto de testes pode ser entendido como o teste final de nosso 
modelo antes de ser liberado no mundo real.
Após termos pré-processado o conjunto de dados, exploraremos diferentes 
técnicas para seleção de recursos para reduzir a dimensionalidade de um 
conjunto de dados.
"""

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 
                   'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 
                   'Proanthocyanins', 'Color intensity', 'Hue','OD280/OD315 of diluted wines', 'Proline']
print('Class labels', np.unique(df_wine['Class label']))
# >>> Class labels [1 2 3]
print(df_wine.head())


"""
As amostras pertencem a uma das três classes diferentes, 1, 2 e 3, que se referem 
aos três tipos diferentes de uvas que foram cultivadas em diferentes regiões 
da Itália.
Uma forma conveniente de particionar aleatoriamente esse conjunto de dados em um conjunto
 de dados de teste e treinamento separado é usar a função train_test_split do 
submódulo cross_validation do scikit-learn:
"""

from sklearn.cross_validation import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


"""
Se estamos dividindo um conjunto de dados em conjuntos de dados de treinamento e teste, 
temos que ter em mente que estamos retendo informações valiosas de que o algoritmo de 
aprendizado poderia se beneficiar. Assim, não queremos alocar muita informação para o 
conjunto de testes. No entanto, quanto menor o conjunto de testes, mais imprecisa será 
a estimativa do erro de generalização. Dividir um conjunto de dados em conjuntos de 
treinamento e teste tem tudo a ver com equilibrar esse compromisso. Na prática, as 
divisões mais usadas são 60:40, 70:30 ou 80:20, dependendo do tamanho do conjunto de 
dados inicial. No entanto, para conjuntos de dados grandes, 90:10 ou 99: 1 se dividem 
em subconjuntos de treinamento e teste também são comuns e apropriados. Em vez de 
descartar os dados de teste alocados após o treinamento e a avaliação do modelo, é 
recomendável treinar novamente um classificador em todo o conjunto de dados para 
obter um desempenho ideal.
"""


