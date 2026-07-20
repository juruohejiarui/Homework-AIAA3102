"""Text, categorical metadata, and shallow numerical feature construction."""
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .normalization import NormalizationConfig, normalize_text

NUMERIC_NAMES = ["char_length","word_count","url_count","mention_count","hashtag_count","punct_count","digit_count","upper_ratio","repeated_punct"]


def shallow_features(df: pd.DataFrame) -> np.ndarray:
    rows=[]
    for value in df.text.fillna("").astype(str):
        letters=sum(c.isalpha() for c in value)
        rows.append([len(value),len(value.split()),len(re.findall(r"https?://|www\.",value,re.I)),len(re.findall(r"@\w+",value)),
                     value.count("#"),len(re.findall(r"[^\w\s]",value)),sum(c.isdigit() for c in value),
                     sum(c.isupper() for c in value)/max(letters,1),len(re.findall(r"[!?.,]{2,}",value))])
    return np.asarray(rows,float)


class FeatureBuilder:
    def __init__(self, mode="text", normalization=NormalizationConfig(), ngram_range=(1,2), min_df=1):
        self.mode, self.normalization = mode, normalization
        # Casing is handled explicitly by normalize_text; disabling vectorizer
        # lowercasing makes the preserve-case lever a genuine single change.
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, sublinear_tf=True, max_features=40000, lowercase=False)
        self.keyword = OneHotEncoder(handle_unknown="ignore")
        self.location = OneHotEncoder(handle_unknown="ignore", min_frequency=2)
        self.scaler = StandardScaler()
        self.fitted_ids: tuple[int,...] = ()

    def _text(self, df): return [normalize_text(x,self.normalization) for x in df.text]
    def fit_transform(self, df: pd.DataFrame):
        self.fitted_ids=tuple(df.id.astype(int))
        parts=[]
        if "text" in self.mode: parts.append(self.vectorizer.fit_transform(self._text(df)))
        if "keyword" in self.mode: parts.append(self.keyword.fit_transform(df[["keyword"]].fillna("<MISSING>")))
        if "location" in self.mode: parts.append(self.location.fit_transform(df[["location"]].fillna("<MISSING>")))
        if "numeric" in self.mode: parts.append(sparse.csr_matrix(self.scaler.fit_transform(shallow_features(df))))
        return sparse.hstack(parts,format="csr") if len(parts)>1 else parts[0]

    def transform(self, df: pd.DataFrame):
        parts=[]
        if "text" in self.mode: parts.append(self.vectorizer.transform(self._text(df)))
        if "keyword" in self.mode: parts.append(self.keyword.transform(df[["keyword"]].fillna("<MISSING>")))
        if "location" in self.mode: parts.append(self.location.transform(df[["location"]].fillna("<MISSING>")))
        if "numeric" in self.mode: parts.append(sparse.csr_matrix(self.scaler.transform(shallow_features(df))))
        return sparse.hstack(parts,format="csr") if len(parts)>1 else parts[0]

    def feature_names(self):
        names=[]
        if "text" in self.mode: names.extend("text:"+x for x in self.vectorizer.get_feature_names_out())
        if "keyword" in self.mode: names.extend("keyword:"+x for x in self.keyword.get_feature_names_out())
        if "location" in self.mode: names.extend("location:"+x for x in self.location.get_feature_names_out())
        if "numeric" in self.mode: names.extend("numeric:"+x for x in NUMERIC_NAMES)
        return np.asarray(names)
