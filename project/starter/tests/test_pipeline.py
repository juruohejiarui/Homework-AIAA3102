import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pipeline.artifacts import PRED_COLS,SUMMARY_COLS,SWEEP_COLS,AUDIT_COLS,validate_artifacts
from pipeline.cli import main
from pipeline.config import DATA,EXPECTED_COUNTS,EXPECTED_POSITIVES,EXPERIMENTS
from pipeline.data import DataValidationError,get_splits,load_data,load_split_ids,validate_and_split
from pipeline.data_quality import duplicate_key,exact_duplicate_groups,conflicting_duplicates,near_duplicate_pairs,validate_dispositions
from pipeline.features import FeatureBuilder
from sklearn.linear_model import LogisticRegression
from pipeline.metrics import binary_metrics,error_transitions,labels_from_scores
from pipeline.normalization import NormalizationConfig,normalize_text

@pytest.fixture(scope="session")
def splits(): return get_splits()

def test_exact_split_row_counts(splits): assert {k:len(v) for k,v in splits.items()}==EXPECTED_COUNTS
def test_exact_positive_counts(splits): assert {k:int(v.target.sum()) for k,v in splits.items()}==EXPECTED_POSITIVES
def test_split_disjointness(splits):
    sets=[set(v.id) for v in splits.values()]; assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
def test_split_union_coverage(splits): assert set().union(*(set(v.id) for v in splits.values()))==set(load_data().id)
def test_id_uniqueness(): assert load_data().id.is_unique
def test_target_validity(): assert set(load_data().target)=={0,1}
def test_missing_split_ids():
    df=load_data(); ids=load_split_ids(); ids["train"][0]=-1
    with pytest.raises(DataValidationError,match="Missing split IDs"): validate_and_split(df,ids)
def test_duplicate_split_ids():
    ids=load_split_ids(); ids["train"][1]=ids["train"][0]
    with pytest.raises(DataValidationError,match="Duplicate"): validate_and_split(load_data(),ids)
def test_invalid_split_file(tmp_path):
    p=tmp_path/"bad.json"; p.write_text("no")
    with pytest.raises(DataValidationError): load_split_ids(p)
def test_url_normalization(): assert normalize_text("x https://a.co/z",NormalizationConfig(url="replace"))=="x urltoken"
def test_url_removal(): assert normalize_text("x https://a.co/z",NormalizationConfig(url="remove"))=="x"
def test_mention_normalization(): assert normalize_text("Hi @Bob",NormalizationConfig(mention="replace"))=="hi usertoken"
def test_mention_removal(): assert normalize_text("Hi @Bob",NormalizationConfig(mention="remove"))=="hi"
def test_hashtag_strip(): assert normalize_text("#Flood",NormalizationConfig(hashtag="strip"))=="flood"
def test_hashtag_remove(): assert normalize_text("x #Flood",NormalizationConfig(hashtag="remove"))=="x"
def test_punctuation_remove(): assert normalize_text("help!!!",NormalizationConfig(punctuation="remove"))=="help"
def test_punctuation_repeat(): assert normalize_text("help!!!",NormalizationConfig(punctuation="repeat"))=="help!"
def test_casing_preserve(): assert normalize_text("HeLP",NormalizationConfig(casing="preserve"))=="HeLP"
def test_casing_lever_reaches_vectorizer():
    d=pd.DataFrame({"id":[1,2],"text":["Fire","fire"]})
    b=FeatureBuilder(normalization=NormalizationConfig(casing="preserve")); b.fit_transform(d)
    assert "Fire" in b.vectorizer.vocabulary_ and "fire" in b.vectorizer.vocabulary_
def test_emoji_remove(): assert normalize_text("fire🔥",NormalizationConfig(emoji="remove"))=="fire"
def test_emoji_replace(): assert "emojitoken" in normalize_text("🔥",NormalizationConfig(emoji="replace"))
def test_missing_text_handling(): assert normalize_text(None)=="" and normalize_text(np.nan)==""
def test_target1_metrics(): assert binary_metrics([0,1,1],[0,1,0])["f1"]==pytest.approx(2/3)
def test_threshold_labels(): assert labels_from_scores([.49,.5]).tolist()==[0,1]
def test_confusion_consistency():
    m=binary_metrics([0,0,1,1],[0,1,0,1]); assert m["tn"]+m["fp"]+m["fn"]+m["tp"]==4
def test_transitions_exclusive_and_cover():
    t=error_transitions(range(4),[0,1,0,1],[1,0,0,1],[0,1,1,0]); assert len(t)==4 and t.category.notna().all()
def test_exact_duplicates():
    d=pd.DataFrame({"id":[1,2],"text":["x","x"],"target":[0,0]}); assert len(exact_duplicate_groups(d))==2
def test_normalized_duplicates():
    d=pd.DataFrame({"id":[1,2],"text":["X!","x"],"target":[0,0]}); assert len(exact_duplicate_groups(d,True))==2
def test_conflicting_duplicates():
    d=pd.DataFrame({"id":[1,2],"text":["X!","x"],"target":[0,1]}); assert len(conflicting_duplicates(d))==2
def test_duplicate_normalization_defined(): assert duplicate_key("Hi @a https://x.co!")=="hi user url"
def test_near_duplicate_validity():
    d=pd.DataFrame({"id":[1,2,3],"text":["large forest fire today","large forest fire today!","hello sunshine"],"target":[1,1,0]}); out=near_duplicate_pairs(d,.5); assert set(out.columns)=={"id","related_id","similarity","label","related_label"} and out.similarity.between(0,1).all()
def test_audit_dispositions():
    validate_dispositions(pd.DataFrame({"disposition":["fix","ambiguous"]}))
    with pytest.raises(ValueError): validate_dispositions(pd.DataFrame({"disposition":["bad"]}))
def test_prediction_schema(): assert set(PRED_COLS)=={"id","y_true","y_pred","score","model_name","ticket"}
def test_summary_schema(): assert "heldout_f1_target_1" in SUMMARY_COLS
def test_sweep_schema(): assert "threshold" in SWEEP_COLS
def test_audit_schema(): assert "disposition" in AUDIT_COLS
def test_vectorizer_train_only(splits):
    b=FeatureBuilder(); b.fit_transform(splits["train"]); assert set(b.fitted_ids)==set(splits["train"].id)
def test_no_dev_leakage(splits):
    b=FeatureBuilder(); b.fit_transform(splits["train"]); assert not set(b.fitted_ids)&set(splits["dev"].id)
def test_no_heldout_leakage(splits):
    b=FeatureBuilder(); b.fit_transform(splits["train"]); assert not set(b.fitted_ids)&set(splits["heldout"].id)
def test_deterministic_vectorization(splits):
    a=FeatureBuilder().fit_transform(splits["train"].head(100)); b=FeatureBuilder().fit_transform(splits["train"].head(100)); assert (a!=b).nnz==0
def test_missing_data_error(tmp_path):
    with pytest.raises(FileNotFoundError): load_data(tmp_path/"missing.csv")
def test_malformed_csv(tmp_path):
    p=tmp_path/"bad.csv"; p.write_text('id,text\n1,"broken')
    with pytest.raises(DataValidationError): load_data(p)
def test_cli_validate(capsys): assert main(["validate-data"])==0
def test_cli_invalid_ticket():
    with pytest.raises(SystemExit): main(["run-ticket","--ticket","9"])
def test_frozen_decisions_exist():
    if (EXPERIMENTS/"decisions.json").exists(): assert all(v["status"]=="frozen" for v in json.loads((EXPERIMENTS/"decisions.json").read_text()).values())
def test_heldout_requires_freeze(tmp_path,monkeypatch):
    import pipeline.cli as cli; monkeypatch.setattr(cli,"EXPERIMENTS",tmp_path)
    with pytest.raises(SystemExit,match="frozen"): cli.main(["run-ticket","--ticket","1","--split","heldout"])
def test_artifacts_if_generated():
    if (Path("results")/"summary.csv").exists(): assert validate_artifacts()["summary_rows"]==5
def test_small_fixture_end_to_end():
    train=pd.DataFrame({"id":[1,2,3,4],"text":["forest fire","happy music","flood warning","nice day"],"target":[1,0,1,0]})
    dev=pd.DataFrame({"id":[5,6],"text":["fire warning","happy day"],"target":[1,0]})
    b=FeatureBuilder(ngram_range=(1,1)); x=b.fit_transform(train); m=LogisticRegression(random_state=3102).fit(x,train.target)
    pred=m.predict(b.transform(dev)); assert pred.tolist()==[1,0] and set(b.fitted_ids)=={1,2,3,4}
def test_score_reconstruction_from_predictions():
    path=Path("predictions/heldout_predictions.csv")
    if path.exists():
        p=pd.read_csv(path); d=json.loads((EXPERIMENTS/"decisions.json").read_text())
        for ticket,part in p.groupby("ticket"):
            threshold=float(d[ticket]["selected_threshold"])
            assert np.array_equal((part.score.to_numpy()>=threshold).astype(int),part.y_pred.to_numpy())
def test_artifact_ordering():
    path=Path("predictions/heldout_predictions.csv")
    if path.exists():
        p=pd.read_csv(path); assert p[["ticket","id"]].values.tolist()==p.sort_values(["ticket","id"])[["ticket","id"]].values.tolist()
