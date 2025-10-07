import os
import csv
from functools import lru_cache

# Funciones auxiliares


@lru_cache(maxsize=1)
def load_product_names(data_dir: str = None):
	"""Return a dict mapping product_id (int) -> product_name (str).

	Reads data/products.csv relative to the repository root by default.
	If the file is missing, returns empty dict.
	"""
	if data_dir is None:
		data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
	path = os.path.join(data_dir, 'products.csv')
	if not os.path.exists(path):
		return {}
	d = {}
	with open(path, newline='', encoding='utf-8') as f:
		reader = csv.DictReader(f)
		for r in reader:
			try:
				pid = int(r.get('product_id') or r.get('productId') or r.get('id'))
			except Exception:
				continue
			name = r.get('product_name') or r.get('name') or ''
			d[pid] = name
	return d