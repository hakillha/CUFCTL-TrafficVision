import re

def natural_sort(l):
	# match integers and convert to actual intergers
	int_value = lambda key: int(re.search('\d+', key).group())
	return sorted(l, key=int_value)