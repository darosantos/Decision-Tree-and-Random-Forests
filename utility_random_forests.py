def get_n_estimators(n_features: int, n_sample: int) -> int:
	"""
		Determina o número de estimadores (arvores) usando o cálculo de arranjo 
		simples considerando o número de atributos a ser analisado
		O uso de arranjo simples se dá pelo motivo de que mos arranjos, os 
		agrupamentos dos elementos dependem da ordem e da natureza dos mesmos.
		Fórmula -> a(n,p) = (n!)/((n-p)!)
		Convém registra que se n_features == n_sample então temos um caso de 
		permutação
	"""
	from math import factorial
	n_features = int(n_features)
	n_sample = int(n_sample)
	n_factorial = factorial(n_features)
	n_p_factorial = factorial(n_features - n_sample)
	arrangement_simple = int(n_factorial/n_p_factorial)
	
	return arrangement_simple

def get_max_depth(n_features: int) -> int:
    from math import sqrt
    from math import log
    max_depth = int(sqrt(n_features) + log(n_features)) + 1
    return max_depth
    

def arrangement_features(features: list,  n_selected: int) -> list:
    from itertools import product
    permsList = list(product(features, repeat=n_selected))
    return permsList

print(arrangement_features(['danilo',  'rodrigues',  'santos'],  2))
print(get_n_estimators(3, 2))
print(len(arrangement_features(['danilo',  'rodrigues',  'santos'],  2)))
