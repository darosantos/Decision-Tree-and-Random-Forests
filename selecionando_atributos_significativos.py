"""
Selecionando atributos significativos
"""

"""
Se notarmos que um modelo tem um desempenho muito melhor em 
um conjunto de dados de treinamento do que no conjunto de dados de teste, 
essa observação é um forte indicador de superajuste. Overfitting significa 
que o modelo ajusta os parâmetros muito de perto às observações específicas 
no conjunto de dados de treinamento, mas não generaliza bem os dados 
reais - dizemos que o modelo tem uma alta variação(variância). Uma razão 
para overfitting é que nosso modelo é muito complexo para os dados de 
treinamento fornecidos e soluções comuns para reduzir o erro de generalização 
são listadas da seguinte forma:
* Colete mais dados de treinamento
* Introduzir uma penalidade por complexidade via regularização
* Escolha um modelo mais simples com menos parâmetros
* Reduza a dimensionalidade dos dados

Coletar mais dados de treinamento geralmente não é aplicável. 
[...] maneiras comuns de reduzir o overfitting pela regularização e 
redução de dimensionalidade por meio da seleção de recursos.
"""

"""
Soluções esparsas com regularização L1
"""

"""
Lembramos do Capítulo 3, Um Tour de Classificadores de Aprendizado de 
Máquina Usando o Scikit-learn, que a regularização L2 é uma abordagem 
para reduzir a complexidade de um modelo penalizando grandes pesos 
individuais, onde definimos a norma L2 do nosso vetor de peso w como 
segue:

L2 = modulo(w)(2,2) = somatorio(j=1, m)w(j,2)

Aqui, simplesmente substituímos o quadrado dos pesos pela soma dos 
valores absolutos dos pesos. Em contraste com a regularização de L2, 
a regularização de L1 produz vetores de recursos esparsos; a maioria 
dos pesos de recursos será zero. A esparsidade pode ser útil na prática 
se tivermos um conjunto de dados de alta dimensão com muitos recursos 
que são irrelevantes, especialmente quando temos dimensões mais 
irrelevantes do que as amostras. Nesse sentido, a regularização de 
L1 pode ser entendida como uma técnica de seleção de características.
Agora, podemos pensar na regularização como adição de um termo de 
penalidade à função de custo para incentivar pesos menores; ou, 
em outras palavras, penalizamos grandes pesos.

Assim, aumentando a força de regularização via regularização
parâmetro, reduzimos os pesos para zero e diminuímos a dependência 
do nosso modelo nos dados de treinamento. 
nosso objetivo é minimizar a soma da função de custo não-penalizada 
mais o termo de penalidade, que pode ser entendido como adição de 
viés e preferindo um modelo mais simples para reduzir a variação 
na ausência de suficiente dados de treinamento para ajustar o modelo.

No entanto, como a penalidade de L1 é a soma dos coeficientes de 
peso absoluto (lembre-se de que o termo L2 é quadrático), podemos 
representá-lo como um orçamento em forma de losango
"""

####### Inserir dados do wine


from sklearn.linear_model import LogisticRegression
# LogisticRegression(penalty='l1')

lr = LogisticRegression(penalty='l1', C=0.1)
lr.fit(X_train_std, y_train)
print('Training accuracy:', lr.score(X_train_std, y_train))
# Training accuracy: 0.983870967742
print('Test accuracy:', lr.score(X_test_std, y_test))
# Test accuracy: 0.981481481481

lr.intercept_

"""
Percebemos que os vetores de peso são esparsos, o que significa que 
eles possuem apenas algumas entradas diferentes de zero. Como 
resultado da regularização L1, que serve como um método para 
seleção de atributos, nós apenas treinamos um modelo que é robusto 
para os recursos potencialmente irrelevantes neste conjunto de dados.
"""
lr.coef_

"""
Por último, vamos traçar o caminho de regularização, que é o 
coeficiente de peso dos diferentes recursos para diferentes 
forças de regularização:
"""
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.subplot(111)
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 
		  'yellow', 'black', 'pink', 'lightgreen', 
		  'lightblue','gray', 'indigo', 'orange']
weights, params = [], []
for c in np.arange(-4, 6):
	lr = LogisticRegression(penalty='l1', C=10**c, random_state=0)
	lr.fit(X_train_std, y_train)
	weights.append(lr.coef_[1])
	params.append(10**c)
weights = np.array(weights)
for column, color in zip(range(weights.shape[1]), colors):
	plt.plot(params, weights[:, column],
	label= df_wine.columns[column+1],color=color)
plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.xscale('log')
plt.legend(loc='upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
plt.show()
