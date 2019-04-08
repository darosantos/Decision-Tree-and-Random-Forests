"""
Avaliando a importância do atributo com florestas aleatórias
"""

"""
Outra abordagem útil para selecionar recursos relevantes de um 
conjunto de dados é usar uma floresta aleatória, uma técnica de 
agrupamento [...]

Usando uma floresta aleatória, podemos medir a importância da característica 
como a redução média de impureza computada de todas as árvores de decisão 
na floresta, sem fazer nenhuma suposição de que nossos dados sejam 
linearmente separáveis ​​ou não. Convenientemente, a implementação de floresta 
aleatória no scikit-learn já coleta importâncias de recursos para nós, para 
que possamos acessá-las através do atributo feature_importances_ depois de 
ajustar um RandomForestClassifier. Ao executar o código a seguir, vamos 
agora treinar uma floresta de 10.000 árvores no dataset Wine e classificar 
os 13 recursos por suas respectivas medidas de importância.

Lembre-se (da nossa discussão no Capítulo 3, Um Tour de Classificadores 
de Aprendizado de Máquina Usando o Scikit-learn) que não precisamos usar 
modelos baseados em árvore padronizados ou normalizados. O código é o seguinte:
"""

from sklearn.ensemble import RandomForestClassifier
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
for f in range(X_train.shape[1]):
	print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))

#1) Alcohol 0.182508
#2) Malic acid 0.158574
#3) Ash 0.150954
#4) Alcalinity of ash 0.131983
#5) Magnesium 0.106564
#6) Total phenols 0.078249
#7) Flavanoids 0.060717
#8) Nonflavanoid phenols 0.032039
#9) Proanthocyanins 0.025385
#10) Color intensity 0.022369
#11) Hue 0.022070
#12) OD280/OD315 of diluted wines 0.014655
#13) Proline 0.013933

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

"""
Depois de executar o código anterior, criamos um gráfico que classifica 
os diferentes recursos no conjunto de dados do Wine por sua importância 
relativa; Observe que as importâncias de recursos são normalizadas, de 
modo que somam 1.0.
"""

"""
Podemos concluir que o teor alcoólico do vinho é a característica mais 
discriminativa no conjunto de dados com base na redução média de impureza 
nas 10.000 árvores de decisão. Curiosamente, os três principais recursos 
classificados no gráfico anterior também estão entre os cinco principais 
recursos na seleção pelo algoritmo SBS que implementamos na seção anterior. 
No entanto, no que diz respeito à interpretabilidade, a técnica de floresta 
aleatória vem com uma pegadinha importante que vale a pena mencionar. Por 
exemplo, se dois ou mais recursos são altamente correlacionados, um recurso 
pode ser classificado muito bem, enquanto as informações do (s) outro (s) 
recurso (s) podem não ser totalmente capturadas. Por outro lado, n
ão precisamos nos preocupar com esse problema se estivermos meramente 
interessados ​​no desempenho preditivo de um modelo, e não na interpretação 
de características importantes. Para concluir esta seção sobre importâncias 
de recursos e florestas aleatórias, vale mencionar que o scikit-learn também 
implementa um método de transformação que seleciona recursos com base em um 
limite especificado pelo usuário após o ajuste do modelo, o que é útil se 
quisermos usar o RandomForestClassifier como seletor de recursos e etapa 
intermediária em um pipeline scikit-learn, que nos permite conectar diferentes
 etapas de pré-processamento com um estimador, como veremos no Capítulo 6, 
 Aprendendo as Melhores Práticas para Avaliação de Modelos e Ajuste de 
 Hyperparameter. Por exemplo, poderíamos definir o limite para 0,15 para 
 reduzir o conjunto de dados para os três recursos mais importantes, Álcool, 
 Ácido Málico e Cinza, usando o seguinte código:
"""

X_selected = forest.transform(X_train, threshold=0.15)
X_selected.shape
## (124, 3)
