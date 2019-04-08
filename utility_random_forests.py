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