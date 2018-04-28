import os

def create_data_dir(out_path):
	try:
		os.makedirs(out_path)
		os.makedirs(os.path.join(out_path,'Annotations'))
		os.makedirs(os.path.join(out_path,'JPEGImages'))
	except OSError, e:
		if e.errno != os.errno.EEXIST:
			raise
		pass