"""Duplicate, near-duplicate, and model-error audit helpers."""
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

DISPOSITIONS = {"fix", "keep_but_flag", "ambiguous", "reject_false_positive"}


def duplicate_key(value: object) -> str:
    text = "" if value is None else str(value).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"@\w+", " user ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_duplicate_groups(df: pd.DataFrame, normalized=False) -> pd.DataFrame:
    key = df.text.fillna("").map(duplicate_key) if normalized else df.text.fillna("").astype(str)
    work = df.assign(_key=key)
    return work[work.duplicated("_key", keep=False) & work._key.ne("")].sort_values(["_key","id"])


def conflicting_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    dup = exact_duplicate_groups(df, normalized=True)
    conflicting = dup.groupby("_key").target.nunique()
    return dup[dup._key.isin(conflicting[conflicting > 1].index)].copy()


def near_duplicate_pairs(df: pd.DataFrame, threshold=.86, limit=100) -> pd.DataFrame:
    texts=df.text.fillna("").astype(str)
    matrix=TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),min_df=2).fit_transform(texts)
    distances, indices=NearestNeighbors(n_neighbors=2,metric="cosine",algorithm="brute").fit(matrix).kneighbors(matrix)
    rows=[]
    for pos,(dist,idx) in enumerate(zip(distances[:,1],indices[:,1])):
        a,b=int(df.iloc[pos].id),int(df.iloc[idx].id)
        if 1-dist>=threshold and a<b:
            rows.append((a,b,float(1-dist),int(df.iloc[pos].target),int(df.iloc[idx].target)))
    return pd.DataFrame(rows,columns=["id","related_id","similarity","label","related_label"]).sort_values(["similarity","id"],ascending=[False,True]).head(limit)


def validate_dispositions(df: pd.DataFrame) -> None:
    invalid=set(df.disposition)-DISPOSITIONS
    if invalid: raise ValueError(f"Invalid audit dispositions: {sorted(invalid)}")

