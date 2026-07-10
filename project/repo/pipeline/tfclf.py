from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

def makePipe(param : dict[str, any]) -> Pipeline :
	return Pipeline([
		("tfidf", TfidfVectorizer(**{
			k.replace("tfidf__", "") : v for k, v in param.items() if k.startswith("tfidf__")
		})),
		("clf", LogisticRegression(**{
			k.replace("clf__", "") : v for k, v in param.items() if k.startswith("clf__")
		}))
	])

