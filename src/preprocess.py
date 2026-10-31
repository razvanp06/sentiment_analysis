import re
import unicodedata

import spacy

EDGE_PUNCT = re.compile(r'^\W+|\W+$')


def identity(x):
	return x


def normalize(text):
	decomposed = unicodedata.normalize('NFD', text)
	return ''.join(c for c in decomposed if unicodedata.category(c) != 'Mn')


def clean_token(word):
	return EDGE_PUNCT.sub('', word)


def load_nlp():
	return spacy.load('ro_core_news_sm', enable=['tok2vec', 'lemmatizer'])


def doc_to_tokens(doc, use_lemma=True, stop_words=None):
	result = []
	for token in doc:
		if not any(c.isalpha() for c in token.text):
			continue
		word = normalize(token.lemma_).lower() if use_lemma else token.lower_
		word = clean_token(word)
		if not word:
			continue
		if stop_words is not None and word in stop_words:
			continue
		result.append(word)
	return result


def predict(text, nlp, vectorizer, model):
	if not text or not text.strip():
		return None
	doc = nlp(normalize(text))
	lemmas = doc_to_tokens(doc, use_lemma=True)
	if not lemmas:
		return None
	X = vectorizer.transform([lemmas])
	proba = model.predict_proba(X)[0]
	return {'negativ': float(proba[0]), 'pozitiv': float(proba[1])}