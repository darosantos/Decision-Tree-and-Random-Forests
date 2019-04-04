import pandas as pd
import numpy as np

"""
Até agora, temos trabalhado apenas com valores numéricos. No entanto, não é incomum 
que conjuntos de dados do mundo real contenham uma ou mais colunas de recursos 
categóricas. Quando estamos falando de dados categóricos, temos que distinguir ainda 
mais entre recursos nominais e ordinais. Recursos ordinais podem ser entendidos como 
valores categóricos que podem ser classificados ou ordenados. Por exemplo, o tamanho 
da camiseta seria um recurso ordinal, porque podemos definir uma ordem XL> L> M. Em 
contraste, os recursos nominais não implicam qualquer ordem e, para continuar com o 
exemplo anterior, poderíamos pensar em T -shirt cor como uma característica nominal, 
uma vez que normalmente não faz sentido dizer que, por exemplo, o vermelho é maior 
que o azul.

Como podemos ver na saída anterior, o DataFrame recém-criado contém uma característica 
nominal (cor), uma característica ordinal (tamanho) e uma coluna de característica 
numérica (preço). Os rótulos de classe (assumindo que criamos um conjunto de dados 
para uma tarefa de aprendizado supervisionada) são armazenados na última coluna. Os 
algoritmos de aprendizado para classificação que discutimos neste livro não usam 
informações ordinais em rótulos de classes.
"""

df = pd.DataFrame([['green', 'M', 10.1, 'class1'],['red', 'L', 13.5, 'class2'],['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']

print(df)

"""
Mapeando atributos ordinais
"""

"""
Para certificar-se de que o algoritmo de aprendizado interpreta os recursos ordinais 
corretamente, precisamos converter os valores das cadeias categóricas em inteiros. 
Infelizmente, não há nenhuma função conveniente que possa derivar automaticamente a ordem 
correta das etiquetas do nosso recurso de tamanho. Assim, temos que definir o mapeamento 
manualmente. No exemplo simples a seguir, vamos supor que sabemos a diferença entre 
os recursos, por exemplo, XL = L + 1 = M + 2.
"""

size_mapping = {'XL': 3, 'L': 2, 'M': 1}
df['size'] = df['size'].map(size_mapping)

print(df)

"""
Se quisermos transformar os valores inteiros de volta para a representação de string 
original em um estágio posterior, podemos simplesmente definir um dicionário de 
mapeamento inverso inv_size_mapping = {v: k for k, v in size_mapping.items()} 
que pode ser usado por meio de o método de mapa dos pandas na coluna de recurso 
transformado semelhante ao dicionário size_mapping que usamos anteriormente.
"""


"""
Codificando rótulos de classes
"""

"""
Muitas bibliotecas de aprendizado de máquina exigem que os rótulos de classe sejam 
codificados como valores inteiros. Embora a maioria dos estimadores para classificação 
em scikit-learn converta rótulos de classe para inteiros internamente, considera-se 
boa prática fornecer rótulos de classe como matrizes inteiras para evitar falhas 
técnicas. Para codificar os rótulos de classe, podemos usar uma abordagem semelhante 
ao mapeamento dos recursos ordinais discutidos anteriormente. Precisamos lembrar que 
os rótulos de classe não são ordinais e não importa qual número inteiro atribuímos a 
um determinado rótulo de sequência. Assim, podemos simplesmente enumerar os rótulos 
de classe começando em 0
"""


class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
print(class_mapping)
df['classlabel'] = df['classlabel'].map(class_mapping)

print(df)


"""
Podemos inverter os pares de valor-chave no dicionário de mapeamento da seguinte 
forma para mapear os rótulos de classe convertidos de volta para a representação 
de sequência original:
"""

inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)

print(df)

"""
Alternativamente, há uma classe LabelEncoder conveniente implementada diretamente 
no scikit-learn para obter o mesmo:
"""

from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)


"""
Note que o método fit_transform é apenas um atalho para chamar fit e transform 
separadamente, e podemos usar o método inverse_transform para transformar os 
rótulos de classes inteiras de volta em sua representação de string original:
"""
y = class_le.inverse_transform(y)
print(y)


"""
Executando one-hot enconding em atributos nominais
"""

"""
Como os estimadores do scikit-learn tratam os rótulos de classe sem qualquer 
ordem, usamos a conveniente classe LabelEncoder para codificar os rótulos de 
string em inteiros. Pode parecer que poderíamos usar uma abordagem semelhante 
para transformar a coluna de cor nominal do nosso conjunto de dados, da seguinte forma:
"""

X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])

print(X)

"""
Se pararmos nesse ponto e alimentarmos o array para nosso classificador, faremos 
um dos erros mais comuns ao lidar com dados categóricos. Você pode identificar o 
problema? Embora os valores das cores não estejam em nenhuma ordem específica, 
um algoritmo de aprendizado agora assumirá que verde é maior que azul e vermelho 
é maior que verde. Embora essa suposição esteja incorreta, o algoritmo ainda 
pode produzir resultados úteis. No entanto, esses resultados não seriam ótimos.

Uma solução comum para esse problema é usar uma técnica chamada codificação 
simples (one-hot encoding). A ideia por trás dessa abordagem é criar 
um novo atributo fictício (dummy feature) para cada valor exclusivo na coluna 
de recurso nominal. Aqui, converteríamos o atributo
de cores em três novos recursos: azul, verde e vermelho. Valores binários podem então 
ser usados para indicar a cor particular de uma amostra; por exemplo, uma amostra 
azul pode ser codificada como azul = 1, verde = 0, vermelho = 0. Para realizar 
essa transformação, podemos usar o OneHotEncoder implementado no módulo 
scikit-learn.preprocessing:

Quando inicializamos o OneHotEncoder, definimos a posição da coluna da variável 
que queremos transformar por meio do parâmetro categorical_features (observe que 
a cor é a primeira coluna na matriz de recursos X). Por padrão, o OneHotEncoder 
retorna uma matriz esparsa quando usamos o método transform e convertemos a 
representação de matriz esparsa em uma matriz NumPy regular (densa) para fins 
de visualização por meio do método toarray. As matrizes esparsas são simplesmente 
uma maneira mais eficiente de armazenar grandes conjuntos de dados e uma que é 
suportada por muitas funções scikit-learn, o que é especialmente útil se contiver 
muitos zeros. Para omitir a etapa toarray, poderíamos inicializar o codificador 
como OneHotEncoder (..., sparse = False) para retornar uma matriz NumPy regular.
"""

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()


"""
Uma maneira ainda mais conveniente de criar esses recursos fictícios por meio 
de uma codificação simples é usar o método get_dummies implementado em pandas. 
Aplicado em um DataFrame, o método get_dummies só converterá colunas de string 
e deixará todas as outras colunas inalteradas:
"""
pd.get_dummies(df[['price', 'color', 'size']])

