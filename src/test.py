import pandas as pd
import spacy

def load_data(path):
	texts = []
	labels = []
	data = pd.read_json(f'{path}')
	

