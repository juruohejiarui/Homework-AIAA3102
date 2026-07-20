"""All five dev-selected, frozen-before-held-out investigation tickets."""
from __future__ import annotations
import json, platform
from dataclasses import asdict
import numpy as np
import pandas as pd
import scipy, sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from . import SEED
from .artifacts import ensure_dirs, write_csv, write_json
from .config import RESULTS,PREDICTIONS,EXPERIMENTS,TICKETS,ROOT
from .data import get_splits
from .data_quality import exact_duplicate_groups,conflicting_duplicates,near_duplicate_pairs,duplicate_key
from .features import FeatureBuilder
from .metrics import binary_metrics,labels_from_scores,error_transitions
from .normalization import NORMALIZATION_CANDIDATES,NormalizationConfig,normalize_text


def fit_model(train, mode="text", normalization=NormalizationConfig(), C=1.0, class_weight=None, family="lr"):
    builder=FeatureBuilder(mode,normalization)
    x=builder.fit_transform(train)
    if family=="svc": model=LinearSVC(C=C,class_weight=class_weight,random_state=SEED,max_iter=5000)
    else: model=LogisticRegression(C=C,class_weight=class_weight,solver="liblinear",random_state=SEED,max_iter=1000)
    model.fit(x,train.target)
    return builder,model


def score(builder,model,df,threshold=.5,family="lr"):
    x=builder.transform(df)
    scores=model.decision_function(x) if family=="svc" else model.predict_proba(x)[:,1]
    cutoff=0.0 if family=="svc" else threshold
    return np.asarray(scores),labels_from_scores(scores,cutoff)


def prediction_frame(df,scores,pred,name,ticket):
    return pd.DataFrame({"id":df.id.astype(int),"y_true":df.target.astype(int),"y_pred":pred.astype(int),"score":scores,"model_name":name,"ticket":ticket})


def decision(ticket,selected,threshold,metric,value,reason,run_id):
    return {"ticket":ticket,"selected_configuration":selected,"selected_threshold":threshold,"selection_metric":metric,
            "dev_result":value,"decision":"accepted","decision_rationale":reason,"run_id":run_id,"status":"frozen","timestamp":"omitted-for-determinism"}


def transition_counts(frame):
    counts=frame.category.value_counts()
    return {k:int(counts.get(k,0)) for k in ("fixed_fp","fixed_fn","new_fp","new_fn")}


def top_coefficients(builder,model,ticket,name,n=30):
    names=builder.feature_names(); coef=model.coef_[0]
    n=min(n,max(1,len(coef)//2))
    idx=np.r_[np.argsort(coef)[:n],np.argsort(coef)[-n:]]
    return pd.DataFrame({"ticket":ticket,"model_name":name,"feature":names[idx],"coefficient":coef[idx],
                         "direction":["negative"]*n+["positive"]*n})


def perturb(df,kind):
    out=df.copy()
    if kind=="keyword_blank": out["keyword"]=""
    elif kind=="keyword_shuffle": out["keyword"]=np.random.default_rng(SEED).permutation(out.keyword.fillna("").to_numpy())
    elif kind=="location_blank": out["location"]=""
    elif kind=="location_shuffle": out["location"]=np.random.default_rng(SEED).permutation(out.location.fillna("").to_numpy())
    else:
        cfg={"url_replace":NormalizationConfig(url="replace"),"mention_replace":NormalizationConfig(mention="replace"),
             "lowercase":NormalizationConfig(),"hashtag_strip":NormalizationConfig(hashtag="strip"),
             "punct_remove":NormalizationConfig(punctuation="remove"),"emoji_remove":NormalizationConfig(emoji="remove")}[kind]
        out["text"]=[normalize_text(x,cfg) for x in out.text]
    return out


def run_all() -> dict:
    ensure_dirs(); split=get_splits(); train,dev,held=split["train"],split["dev"],split["heldout"]
    decisions=[]; registry=[]; summaries=[]; all_held=[]; all_trans=[]; confusions=[]; tops=[]; stress=[]
    decision_path=EXPERIMENTS/"decisions.json"
    write_json({},decision_path)
    def persist_decisions():
        write_json({d["ticket"]:d for d in decisions},decision_path)

    # Ticket 1: assignment-driven baseline; freeze before first held-out scoring.
    bld,mdl=fit_model(train); ds,dp=score(bld,mdl,dev); dm=binary_metrics(dev.target,dp)
    bname="tfidf_12_sublinear_lr_liblinear_c1"; ticket="ticket-1"
    decisions.append(decision(ticket,{"mode":"text","normalization":asdict(NormalizationConfig()),"vectorizer":bld.vectorizer.get_params(),
        "classifier":mdl.get_params()},.5,"dev_f1_target_1",dm["f1"],"Reasonable text-only baseline fixed from the assignment and dev evaluation.","t1-baseline"))
    persist_decisions()
    hs,hp=score(bld,mdl,held); hm=binary_metrics(held.target,hp)
    base_dev_pred,base_held_pred=dp.copy(),hp.copy(); base_dev_scores=ds.copy()
    write_csv(prediction_frame(dev,ds,dp,bname,ticket),PREDICTIONS/"dev"/"ticket-1.csv",["id"])
    all_held.append(prediction_frame(held,hs,hp,bname,ticket)); tops.append(top_coefficients(bld,mdl,ticket,bname))
    registry.append((ticket,bname,"baseline",dm["f1"],hm["f1"],.5))
    confusions.extend([(ticket,"dev",bname,*[dm[k] for k in ("tn","fp","fn","tp")]),(ticket,"heldout",bname,*[hm[k] for k in ("tn","fp","fn","tp")])])
    summaries.append((ticket,bname,dm["f1"],hm["f1"],hm["accuracy"],0,0,0,0,"accepted","Assignment-driven baseline frozen at threshold 0.5"))
    floor=np.zeros(len(dev),int); fm=binary_metrics(dev.target,floor); floor_h=binary_metrics(held.target,np.zeros(len(held),int))
    registry.append((ticket,"majority_floor","floor",fm["f1"],floor_h["f1"],.5)); discrepancy=[("submitted_baseline","none","frozen configuration",dm["f1"],hm["f1"],hm["f1"]-.7574221578566256,int(mdl.n_iter_[0]))]
    for label,cfg in [("unigrams_only",{"ngram_range":(1,1)}),("lbfgs_solver",{})]:
        pb=FeatureBuilder("text",NormalizationConfig(),**cfg); x=pb.fit_transform(train)
        pm=LogisticRegression(solver="lbfgs" if label=="lbfgs_solver" else "liblinear",random_state=SEED,max_iter=1000).fit(x,train.target)
        ps,pp=score(pb,pm,dev); phs,php=score(pb,pm,held); pdm=binary_metrics(dev.target,pp); phm=binary_metrics(held.target,php)
        registry.append((ticket,label,"diagnostic_probe",pdm["f1"],phm["f1"],.5)); discrepancy.append((label,"ngram_range" if label=="unigrams_only" else "solver",str(cfg or {"solver":"lbfgs"}),pdm["f1"],phm["f1"],phm["f1"]-.7574221578566256,int(pm.n_iter_[0])))

    # Ticket 2: each candidate changes one normalization variable on dev.
    norm_runs=[]
    for name,cfg in NORMALIZATION_CANDIDATES.items():
        cb,cm=fit_model(train,normalization=cfg); cs,cp=score(cb,cm,dev); met=binary_metrics(dev.target,cp)
        trans=error_transitions(dev.id,dev.target,base_dev_pred,cp,bname,name); trans["ticket"]=ticket2="ticket-2"; all_trans.append(trans)
        norm_runs.append((met["f1"],name,cfg,cb,cm,cs,cp,met)); registry.append((ticket2,name,"single_normalization_lever",met["f1"],np.nan,.5))
    norm_runs.sort(key=lambda x:(-x[0],x[1])); _,nname,ncfg,nb,nm,nds,ndp,ndm=norm_runs[0]
    decisions.append(decision(ticket2,{"candidate":nname,"normalization":asdict(ncfg)},.5,"dev_f1_target_1",ndm["f1"],"Highest dev F1 among controlled single-lever candidates; mechanism checked via transitions.","t2-normalization"))
    persist_decisions()
    nhs,nhp=score(nb,nm,held); nhm=binary_metrics(held.target,nhp); nt=error_transitions(held.id,held.target,base_held_pred,nhp,bname,nname); nt["ticket"]=ticket2; all_trans.append(nt); nc=transition_counts(nt)
    write_csv(prediction_frame(dev,nds,ndp,nname,ticket2),PREDICTIONS/"dev"/"ticket-2.csv",["id"]); all_held.append(prediction_frame(held,nhs,nhp,nname,ticket2)); tops.append(top_coefficients(nb,nm,ticket2,nname))
    summaries.append((ticket2,nname,ndm["f1"],nhm["f1"],nhm["accuracy"],*[nc[k] for k in ("fixed_fp","fixed_fn","new_fp","new_fn")],"accepted",f"Dev-selected single lever: {nname}"))
    confusions.extend([(ticket2,"dev",nname,*[ndm[k] for k in ("tn","fp","fn","tp")]),(ticket2,"heldout",nname,*[nhm[k] for k in ("tn","fp","fn","tp")])])
    for kind in ("url_replace","mention_replace","lowercase","hashtag_strip","punct_remove","emoji_remove"):
        ps,pp=score(bld,mdl,perturb(dev,kind)); pm=binary_metrics(dev.target,pp); tr=error_transitions(dev.id,dev.target,base_dev_pred,pp); c=transition_counts(tr)
        stress.append((ticket2,kind,dm["f1"],pm["f1"],pm["f1"]-dm["f1"],(pm["f1"]-dm["f1"])/dm["f1"],int((pp!=base_dev_pred).sum()),*[c[k] for k in ("fixed_fp","fixed_fn","new_fp","new_fn")],"superficial-text robustness probe"))

    # Ticket 3: controlled signal ablations/additions.
    modes=["text","keyword","location","numeric","keyword+location","text+keyword","text+numeric"]
    feat_runs=[]
    for mode in modes:
        fb,fm_=fit_model(train,mode=mode); fs,fp=score(fb,fm_,dev); met=binary_metrics(dev.target,fp); name=mode.replace("+","_plus_")
        feat_runs.append((met["f1"],name,mode,fb,fm_,fs,fp,met)); registry.append(("ticket-3",name,"feature_ablation",met["f1"],np.nan,.5))
        tr=error_transitions(dev.id,dev.target,base_dev_pred,fp,bname,name); tr["ticket"]="ticket-3"; all_trans.append(tr)
        if hasattr(fm_,"coef_"): tops.append(top_coefficients(fb,fm_,"ticket-3",name,15))
    feat_runs.sort(key=lambda x:(-x[0],x[1])); _,fname,fmode,fb,fm_,fds,fdp,fdm=feat_runs[0]
    decisions.append(decision("ticket-3",{"mode":fmode},.5,"dev_f1_target_1",fdm["f1"],"Best controlled dev feature set; shortcut-only results retained for audit.","t3-features"))
    persist_decisions()
    fhs,fhp=score(fb,fm_,held); fhm=binary_metrics(held.target,fhp); ft=error_transitions(held.id,held.target,base_held_pred,fhp,bname,fname); ft["ticket"]="ticket-3"; all_trans.append(ft); fc=transition_counts(ft)
    write_csv(prediction_frame(dev,fds,fdp,fname,"ticket-3"),PREDICTIONS/"dev"/"ticket-3.csv",["id"]); all_held.append(prediction_frame(held,fhs,fhp,fname,"ticket-3"))
    summaries.append(("ticket-3",fname,fdm["f1"],fhm["f1"],fhm["accuracy"],*[fc[k] for k in ("fixed_fp","fixed_fn","new_fp","new_fn")],"accepted",f"Dev-selected feature set: {fmode}"))
    confusions.extend([("ticket-3","dev",fname,*[fdm[k] for k in ("tn","fp","fn","tp")]),("ticket-3","heldout",fname,*[fhm[k] for k in ("tn","fp","fn","tp")])])
    for kind in ("keyword_blank","keyword_shuffle","location_blank","location_shuffle"):
        ps,pp=score(fb,fm_,perturb(dev,kind)); pm=binary_metrics(dev.target,pp); tr=error_transitions(dev.id,dev.target,fdp,pp); c=transition_counts(tr)
        stress.append(("ticket-3",kind,fdm["f1"],pm["f1"],pm["f1"]-fdm["f1"],(pm["f1"]-fdm["f1"])/max(fdm["f1"],1e-12),int((pp!=fdp).sum()),*[c[k] for k in ("fixed_fp","fixed_fn","new_fp","new_fn")],"metadata shortcut probe"))

    # Ticket 4: interpretable LR grid, threshold sweep, and LinearSVC comparison.
    sweep=[]
    for threshold in np.round(np.arange(.10,.901,.01),2):
        tm=binary_metrics(dev.target,labels_from_scores(base_dev_scores,threshold)); sweep.append(("ticket-4",threshold,tm["precision"],tm["recall"],tm["f1"],tm["tp"],tm["fp"],tm["tn"],tm["fn"],tm["predicted_positive"]))
    candidates=[]
    for C in (.1,.25,.5,1,2,4,10):
      for weight in (None,"balanced"):
        tb,tm_=fit_model(train,C=C,class_weight=weight); ts,_=score(tb,tm_,dev)
        best=max((binary_metrics(dev.target,labels_from_scores(ts,t))["f1"],float(t)) for t in np.round(np.arange(.2,.801,.01),2))
        tp=labels_from_scores(ts,best[1]); met=binary_metrics(dev.target,tp); name=f"lr_c{C}_{weight or 'none'}_t{best[1]:.2f}"
        candidates.append((met["f1"],name,"lr",C,weight,best[1],tb,tm_,ts,tp,met)); registry.append(("ticket-4",name,"model_threshold",met["f1"],np.nan,best[1]))
    sb,sm=fit_model(train,family="svc"); ss,sp=score(sb,sm,dev,family="svc"); svm=binary_metrics(dev.target,sp); candidates.append((svm["f1"],"linear_svc_c1","svc",1,None,0.0,sb,sm,ss,sp,svm)); registry.append(("ticket-4","linear_svc_c1","alternative_classifier",svm["f1"],np.nan,0.0))
    candidates.sort(key=lambda x:(-x[0],x[1])); _,mname,family,C,weight,threshold,mb,mm_,mds,mdp,mdm=candidates[0]
    decisions.append(decision("ticket-4",{"family":family,"C":C,"class_weight":weight,"score_type":"decision_function" if family=="svc" else "probability"},threshold,"dev_f1_target_1",mdm["f1"],"Best dev operating point in limited interpretable search.","t4-decision-rule"))
    persist_decisions()
    mhs,mhp=score(mb,mm_,held,threshold,family); mhm=binary_metrics(held.target,mhp); mt=error_transitions(held.id,held.target,base_held_pred,mhp,bname,mname); mt["ticket"]="ticket-4"; all_trans.append(mt); mc=transition_counts(mt)
    write_csv(prediction_frame(dev,mds,mdp,mname,"ticket-4"),PREDICTIONS/"dev"/"ticket-4.csv",["id"]); all_held.append(prediction_frame(held,mhs,mhp,mname,"ticket-4"))
    summaries.append(("ticket-4",mname,mdm["f1"],mhm["f1"],mhm["accuracy"],*[mc[k] for k in ("fixed_fp","fixed_fn","new_fp","new_fn")],"accepted","Dev-selected model and operating threshold"))
    confusions.extend([("ticket-4","dev",mname,*[mdm[k] for k in ("tn","fp","fn","tp")]),("ticket-4","heldout",mname,*[mhm[k] for k in ("tn","fp","fn","tp")])])

    # Ticket 5: evidence audit only; labels remain untouched and model is inherited frozen.
    full=pd.concat([train.assign(split="train"),dev.assign(split="dev"),held.assign(split="heldout")]).sort_values("id")
    exact=exact_duplicate_groups(full); normal=exact_duplicate_groups(full,True); conflict=conflicting_duplicates(full); near=near_duplicate_pairs(full)
    audit=[]
    for issue,frame in (("exact_duplicate",exact),("normalized_duplicate",normal),("conflicting_label_duplicate",conflict)):
        for _,row in frame.iterrows():
            related=frame[(frame._key==row["_key"])&(frame.id!=row.id)].id.astype(str).tolist()
            disposition="ambiguous" if issue=="conflicting_label_duplicate" else "keep_but_flag"
            audit.append((int(row.id),issue,f"related_ids={','.join(related[:10])}; normalized_key={duplicate_key(row.text)[:100]}",disposition,.95,row["split"],int(row.target),"",",".join(related),"duplicate evidence"))
    for row in near.itertuples():
        audit.append((int(row.id),"near_duplicate",f"related_id={row.related_id}; cosine={row.similarity:.4f}","ambiguous" if row.label!=row.related_label else "keep_but_flag",float(row.similarity),"",int(row.label),"",str(row.related_id),"character 3-5 gram TF-IDF cosine"))
    # High-confidence held-out errors: analysis, never automatic relabels.
    for pos,row in held.reset_index(drop=True).iterrows():
        if mhp[pos]!=row.target and (mhs[pos]>.8 or mhs[pos]<.2):
            typ="high_confidence_fp" if mhp[pos]==1 else "high_confidence_fn"
            audit.append((int(row.id),typ,f"model_score={mhs[pos]:.4f}; text={str(row.text)[:140]}","ambiguous",float(max(mhs[pos],1-mhs[pos])),"heldout",int(row.target),"","","model disagreement is not label proof"))
    adf=pd.DataFrame(audit,columns=["id","issue_type","evidence","disposition","confidence","split","original_label","proposed_label","related_ids","rationale"]).drop_duplicates(["id","issue_type","evidence"])
    if adf.empty: adf=pd.DataFrame([(int(full.iloc[0].id),"no_issue_sentinel","No auditable issue found","reject_false_positive",0.0,"train",int(full.iloc[0].target),"","","schema sentinel")],columns=adf.columns)
    decisions.append(decision("ticket-5",{"label_correction":"none","audit_model":mname},threshold,"dev_f1_target_1",mdm["f1"],"Audit evidence was insufficient for an automatic training-label correction.","t5-data-quality"))
    persist_decisions()
    t5=error_transitions(held.id,held.target,base_held_pred,mhp,bname,mname); t5["ticket"]="ticket-5"; all_trans.append(t5); t5c=transition_counts(t5)
    all_held.append(prediction_frame(held,mhs,mhp,mname,"ticket-5")); write_csv(prediction_frame(dev,mds,mdp,mname,"ticket-5"),PREDICTIONS/"dev"/"ticket-5.csv",["id"])
    summaries.append(("ticket-5",mname,mdm["f1"],mhm["f1"],mhm["accuracy"],*[t5c[k] for k in ("fixed_fp","fixed_fn","new_fp","new_fn")],"keep_original_labels","Audit-only: no labels modified"))
    confusions.extend([("ticket-5","dev",mname,*[mdm[k] for k in ("tn","fp","fn","tp")]),("ticket-5","heldout",mname,*[mhm[k] for k in ("tn","fp","fn","tp")])])

    # Stable artifact emission.
    persist_decisions()
    write_csv(pd.DataFrame(registry,columns=["ticket","model_name","experiment_type","dev_f1_target_1","heldout_f1_target_1","threshold"]),RESULTS/"experiment_registry.csv",["ticket","model_name"])
    write_csv(pd.DataFrame(discrepancy,columns=["probe","single_factor","setting","dev_f1_target_1","heldout_f1_target_1","difference_from_reference","iterations"]),RESULTS/"discrepancy_comparison.csv",["probe"])
    write_csv(pd.concat(all_held),PREDICTIONS/"heldout_predictions.csv",["ticket","id"])
    write_csv(pd.DataFrame(summaries,columns=["ticket","model_name","dev_f1_target_1","heldout_f1_target_1","heldout_accuracy","fixed_fp","fixed_fn","new_fp","new_fn","decision","decision_reason"]),RESULTS/"summary.csv",["ticket"])
    write_csv(pd.DataFrame(sweep,columns=["ticket","threshold","precision_target_1","recall_target_1","f1_target_1","tp","fp","tn","fn","predicted_positive_count"]),RESULTS/"threshold_sweep.csv",["threshold"])
    write_csv(adf,RESULTS/"data_quality_audit.csv",["id","issue_type"])
    write_csv(pd.concat(all_trans),RESULTS/"error_transitions.csv",["ticket","id","candidate_model"])
    write_csv(pd.DataFrame(stress,columns=["ticket","perturbation","baseline_metric","perturbed_metric","absolute_change","relative_change","prediction_flips","fixed_fp","fixed_fn","new_fp","new_fn","interpretation"]),RESULTS/"perturbation_stress.csv",["ticket","perturbation"])
    write_csv(pd.DataFrame(confusions,columns=["ticket","split","model_name","tn","fp","fn","tp"]),RESULTS/"confusion_matrices.csv",["ticket","split"])
    write_csv(pd.concat(tops),RESULTS/"top_features.csv",["ticket","model_name","direction","coefficient"])
    write_json({"python":platform.python_version(),"scikit_learn":sklearn.__version__,"pandas":pd.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"seed":SEED,"cpu_only":True},RESULTS/"environment.json")
    summary_columns=["ticket","model_name","dev_f1_target_1","heldout_f1_target_1","heldout_accuracy","fixed_fp","fixed_fn","new_fp","new_fn","decision","decision_reason"]
    return {"summary":pd.DataFrame(summaries,columns=summary_columns),"floor":fm,"audit_rows":len(adf),"decisions":decisions}
