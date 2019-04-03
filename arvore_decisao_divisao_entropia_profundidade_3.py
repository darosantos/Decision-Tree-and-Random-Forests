"""
 Usando o scikit-learn, vamos agora treinar uma árvore de decisão com uma profundidade máxima de 3 usando a entropia como um critério para a impureza
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(X_train, y_train)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

plot_decision_regions(X_combined, y_combined, classifier=tree, test_idx=range(105,150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()

"""
Um recurso interessante no scikit-learn é que ele nos permite exportar o arquivo de decisão como um arquivo .dot
após o treinamento, que podemos visualizar usando o programa GraphViz
Primeiro, criamos o arquivo .dot via scikit-learn usando a função export_graphviz do submódulo da árvore, da seguinte maneira:
"""
export_graphviz(tree, out_file='tree.dot', feature_names=['petal length', 'petal width'])
