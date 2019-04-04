import pandas as pd


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

Para certificar-se de que o algoritmo de aprendizado interpreta os recursos ordinais 
corretamente, precisamos converter os valores das cadeias categóricas em inteiros. 
Infelizmente, não há nenhuma função conveniente que possa derivar automaticamente a ordem 
correta das etiquetas do nosso recurso de tamanho. Assim, temos que definir o mapeamento 
manualmente. No exemplo simples a seguir, vamos supor que sabemos a diferença entre 
os recursos, por exemplo, XL = L + 1 = M + 2.

"""

df = pd.DataFrame([['green', 'M', 10.1, 'class1'],['red', 'L', 13.5, 'class2'],['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']

print(df)

size_mapping = {'XL': 3, 'L': 2, 'M': 1}
df['size'] = df['size'].map(size_mapping)

print(df)

