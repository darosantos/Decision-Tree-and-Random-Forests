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