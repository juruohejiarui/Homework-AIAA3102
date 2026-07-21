"""All five dev-selected, frozen-before-held-out investigation tickets."""
from __future__ import annotations
import json, platform
from dataclasses import asdict
import numpy as np
import pandas as pd
import scipy, sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.svm import LinearSVC
from scipy.stats import pearsonr, spearmanr
from . import SEED
from .artifacts import ensure_dirs, write_csv, write_json
from .config import RESULTS,PREDICTIONS,EXPERIMENTS,TICKETS,ROOT,PENDING_DECISIONS
from .data import get_splits, get_train_dev
from .data_quality import exact_duplicate_groups,conflicting_duplicates,near_duplicate_pairs,duplicate_key
from .features import FeatureBuilder
from .metrics import binary_metrics,labels_from_scores,error_transitions
from .normalization import NORMALIZATION_CANDIDATES,NormalizationConfig,normalize_text


def fit_model(train, mode="text", normalization=NormalizationConfig(), C=1.0, class_weight=None, family="lr",
              seed=SEED, solver="liblinear", **vectorizer_kwargs):
    builder=FeatureBuilder(mode,normalization,**vectorizer_kwargs)
    x=builder.fit_transform(train)
    if family=="svc": model=LinearSVC(C=C,class_weight=class_weight,random_state=seed,max_iter=5000)
    else: model=LogisticRegression(C=C,class_weight=class_weight,solver=solver,random_state=seed,max_iter=1000)
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
            "dev_result":value,"decision":"accepted","decision_rationale":reason,"run_id":run_id,"status":"pending","timestamp":"omitted-for-determinism"}


TICKET1_PROBE_POLICY={
    "matrix":"fixed one-factor diagnostic matrix defined in pipeline.experiments",
    "freeze_boundary":"The submitted baseline is frozen before any diagnostic held-out score is computed.",
    "heldout_use":"Held-out probe scores are forensic evidence only; they cannot replace the baseline or select later ticket settings.",
    "interpretation":"Compare dev and held-out deltas for agreement; do not rank probes by held-out F1.",
    "representative_transition_probes":"c_10 and no_vector_normalization are fixed after dev comparison and before held-out evaluation."
}
TICKET1_TRANSITION_PROBES={"c_10","no_vector_normalization"}


def transition_counts(frame):
    counts=frame.category.value_counts()
    return {k:int(counts.get(k,0)) for k in ("fixed_fp","fixed_fn","new_fp","new_fn")}


def append_stress(rows, ticket, model_name, builder, model, reference, reference_pred, perturbation, interpretation):
    _, perturbed_pred=score(builder,model,perturb(reference,perturbation))
    reference_metrics=binary_metrics(reference.target,reference_pred)
    perturbed_metrics=binary_metrics(reference.target,perturbed_pred)
    transitions=error_transitions(reference.id,reference.target,reference_pred,perturbed_pred)
    counts=transition_counts(transitions)
    rows.append((ticket,model_name,perturbation,reference_metrics["f1"],perturbed_metrics["f1"],
                 perturbed_metrics["f1"]-reference_metrics["f1"],
                 (perturbed_metrics["f1"]-reference_metrics["f1"])/max(reference_metrics["f1"],1e-12),
                 int((perturbed_pred!=reference_pred).sum()),
                 *[counts[key] for key in ("fixed_fp","fixed_fn","new_fp","new_fn")],interpretation))


def top_coefficients(builder,model,ticket,name,n=30):
    names=builder.feature_names(); coef=model.coef_[0]
    n=min(n,max(1,len(coef)//2))
    idx=np.r_[np.argsort(coef)[:n],np.argsort(coef)[-n:]]
    return pd.DataFrame({"ticket":ticket,"model_name":name,"feature":names[idx],"coefficient":coef[idx],
                         "direction":["negative"]*n+["positive"]*n})


def write_ticket4_decision_curves(y_true, scores, selected_threshold):
    """Write dev-only curves that explain the already selected Ticket 4 operating point."""
    thresholds=np.round(np.arange(.20,.801,.01),2)
    rows=[]
    for threshold in thresholds:
        metrics=binary_metrics(y_true,labels_from_scores(scores,threshold))
        rows.append(("dev","lr_c10_balanced",threshold,metrics["precision"],metrics["recall"],metrics["f1"]))
    curve=pd.DataFrame(rows,columns=["split","model_name","threshold","precision_target_1","recall_target_1","f1_target_1"])
    write_csv(curve,RESULTS/"ticket4_dev_decision_curve.csv",["threshold"])

    precision,recall,pr_thresholds=precision_recall_curve(y_true,scores)
    pr=pd.DataFrame({"split":"dev","model_name":"lr_c10_balanced","threshold":np.r_[pr_thresholds,np.nan],
                     "precision_target_1":precision,"recall_target_1":recall})
    write_csv(pr,RESULTS/"ticket4_dev_precision_recall_curve.csv")

    figures=RESULTS/"figures"
    figures.mkdir(parents=True,exist_ok=True)
    selected=curve[curve.threshold==selected_threshold].iloc[0]
    plt.figure(figsize=(5.8,4.0))
    plt.plot(recall,precision,color="#15616d",linewidth=2)
    plt.scatter([selected.recall_target_1],[selected.precision_target_1],color="#c1121f",zorder=3,
                label=f"Selected threshold = {selected_threshold:.2f}")
    plt.xlabel("Recall (target=1)")
    plt.ylabel("Precision (target=1)")
    plt.title("Ticket 4 Dev Precision-Recall Curve")
    plt.grid(alpha=.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figures/"ticket4_dev_precision_recall.png",dpi=180)
    plt.close()

    plt.figure(figsize=(5.8,4.0))
    plt.plot(curve.threshold,curve.f1_target_1,color="#15616d",linewidth=2)
    plt.axvline(selected_threshold,color="#c1121f",linestyle="--",linewidth=1.5)
    plt.scatter([selected_threshold],[selected.f1_target_1],color="#c1121f",zorder=3,
                label=f"Selected threshold = {selected_threshold:.2f}")
    plt.xlabel("Probability threshold")
    plt.ylabel("F1 (target=1)")
    plt.title("Ticket 4 Dev F1 by Threshold")
    plt.grid(alpha=.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(figures/"ticket4_dev_f1_threshold.png",dpi=180)
    plt.close()


def figures_path(filename):
    figures=RESULTS/"figures"
    figures.mkdir(parents=True,exist_ok=True)
    return figures/filename


def write_ticket1_probe_agreement_figure(discrepancy):
    probes=discrepancy[discrepancy.probe!="submitted_baseline"].copy()
    fig,axes=plt.subplots(1,2,figsize=(10.2,5.4),gridspec_kw={"width_ratios":[1.3,1]})
    focus_minimum,focus_maximum=-.06,.035
    full_minimum=min(probes.dev_delta_from_baseline.min(),probes.heldout_delta_from_baseline.min())-.02
    full_maximum=max(probes.dev_delta_from_baseline.max(),probes.heldout_delta_from_baseline.max())+.02
    for axis,minimum,maximum,title in [
        (axes[0],focus_minimum,focus_maximum,"Focused near-zero deltas"),
        (axes[1],full_minimum,full_maximum,"Full delta range"),
    ]:
        for factor,part in probes.groupby("single_factor",sort=True):
            axis.scatter(part.dev_delta_from_baseline,part.heldout_delta_from_baseline,s=34,alpha=.8,label=factor)
        axis.plot([minimum,maximum],[minimum,maximum],color="#555555",linestyle="--",linewidth=1,label="equal dev/held-out delta")
        axis.axhline(0,color="#aaaaaa",linewidth=.8)
        axis.axvline(0,color="#aaaaaa",linewidth=.8)
        axis.set(xlim=(minimum,maximum),ylim=(minimum,maximum),title=title)
        axis.grid(alpha=.2)
    label_offsets={"c_10":(5,-12),"c_4":(5,6),"no_vector_normalization":(5,6),"unigrams_only":(-48,-10),"l1_normalization":(5,5)}
    for label in ("c_10","c_4","no_vector_normalization","unigrams_only","l1_normalization"):
        row=probes[probes.probe==label].iloc[0]
        for axis in axes:
            if axis is axes[1] or (focus_minimum<=row.dev_delta_from_baseline<=focus_maximum and focus_minimum<=row.heldout_delta_from_baseline<=focus_maximum):
                axis.annotate(label,(row.dev_delta_from_baseline,row.heldout_delta_from_baseline),xytext=label_offsets[label],textcoords="offset points",fontsize=7)
    axes[0].set(xlabel="Dev F1 delta from frozen baseline",ylabel="Held-out F1 delta from frozen baseline")
    axes[1].set(xlabel="Dev F1 delta from frozen baseline")
    handles,labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,labels,fontsize=6,ncol=4,frameon=False,loc="lower center",bbox_to_anchor=(.5,.01))
    fig.suptitle("Ticket 1 Frozen Probe Agreement",y=.99)
    fig.tight_layout(rect=(0,.16,1,.93))
    fig.savefig(figures_path("ticket1_probe_delta_agreement.png"),dpi=180)
    plt.close(fig)


def write_ticket2_evidence_artifacts(candidates, stress):
    write_csv(candidates,RESULTS/"ticket2_normalization_comparison.csv",["rank","candidate"])
    fig,axes=plt.subplots(1,2,figsize=(10.0,4.6),gridspec_kw={"width_ratios":[1.15,1]})
    ranked=candidates.sort_values(["dev_f1_target_1","candidate"])
    colors=["#c1121f" if selected else "#15616d" for selected in ranked.selected]
    plot_minimum=ranked.dev_f1_target_1.min()-.001
    plot_maximum=ranked.dev_f1_target_1.max()+.001
    axes[0].hlines(ranked.candidate,plot_minimum,ranked.dev_f1_target_1,color="#a7b6b8",linewidth=1.2)
    axes[0].scatter(ranked.dev_f1_target_1,ranked.candidate,color=colors,s=52,zorder=3)
    raw=float(candidates.loc[candidates.candidate=="raw","dev_f1_target_1"].iloc[0])
    axes[0].axvline(raw,color="#555555",linestyle="--",linewidth=1,label="raw baseline")
    axes[0].set(xlim=(plot_minimum,plot_maximum),xlabel="Dev F1 (target=1)",title="Ticket 2: One-Lever Normalization Candidates")
    axes[0].legend(frameon=False,fontsize=8)
    axes[0].grid(axis="x",alpha=.2)

    order=["url_replace","mention_replace","lowercase","hashtag_strip","punct_remove","emoji_remove"]
    selected=stress[(stress.ticket=="ticket-2") & stress.perturbation.isin(order)].copy()
    model_labels={"tfidf_12_sublinear_lr_liblinear_c1":"raw baseline","url_replace":"URL-replacement model"}
    x=np.arange(len(order)); width=.36
    for index,(model_name,label) in enumerate(model_labels.items()):
        values=selected[selected.model_name==model_name].set_index("perturbation").reindex(order).absolute_change
        axes[1].bar(x+(index-.5)*width,values,width,label=label)
    axes[1].axhline(0,color="#555555",linewidth=.8)
    axes[1].set(xticks=x,xticklabels=order,ylabel="Dev F1 change after perturbation",title="Ticket 2: Surface-Signal Stress")
    axes[1].tick_params(axis="x",rotation=45,labelsize=8)
    axes[1].legend(frameon=False,fontsize=7)
    axes[1].grid(axis="y",alpha=.2)
    fig.tight_layout()
    fig.savefig(figures_path("ticket2_normalization_and_stress.png"),dpi=180)
    plt.close(fig)


def write_ticket3_evidence_artifacts(candidates, stress):
    write_csv(candidates,RESULTS/"ticket3_feature_comparison.csv",["rank","feature_mode"])
    fig,axes=plt.subplots(1,2,figsize=(10.2,4.7),gridspec_kw={"width_ratios":[1,1.2]})
    ranked=candidates.sort_values(["dev_f1_target_1","feature_mode"])
    colors=["#c1121f" if selected else "#15616d" for selected in ranked.selected]
    axes[0].barh(ranked.feature_mode,ranked.dev_f1_target_1,color=colors)
    axes[0].set(xlabel="Dev F1 (target=1)",title="Ticket 3: Feature-Source Ablation")
    axes[0].grid(axis="x",alpha=.2)

    metadata=stress[stress.ticket=="ticket-3"].copy()
    metadata["label"]=metadata.model_name.str.replace("_plus_","+",regex=False)+"\n"+metadata.perturbation.str.replace("_"," ",regex=False)
    metadata=metadata.sort_values(["model_name","perturbation"])
    colors=["#c1121f" if "blank" in label else "#e67e22" for label in metadata.perturbation]
    axes[1].barh(metadata.label,metadata.absolute_change,color=colors)
    axes[1].axvline(0,color="#555555",linewidth=.8)
    axes[1].set(xlabel="Dev F1 change after metadata intervention",title="Ticket 3: Metadata Reliance Stress")
    axes[1].tick_params(axis="y",labelsize=7)
    axes[1].grid(axis="x",alpha=.2)
    fig.tight_layout()
    fig.savefig(figures_path("ticket3_feature_and_stress.png"),dpi=180)
    plt.close(fig)


def write_ticket4_grid_artifacts(candidates):
    write_csv(candidates,RESULTS/"ticket4_model_grid.csv",["dev_f1_target_1","model_name"])
    lr=candidates[candidates.family=="lr"].copy()
    pivot=lr.pivot(index="C",columns="class_weight",values="dev_f1_target_1").sort_index()
    fig,axis=plt.subplots(figsize=(5.7,4.1))
    image=axis.imshow(pivot.to_numpy(),cmap="YlGnBu",aspect="auto")
    axis.set(xticks=np.arange(len(pivot.columns)),xticklabels=pivot.columns,yticks=np.arange(len(pivot.index)),yticklabels=[f"{value:g}" for value in pivot.index],xlabel="Class weighting",ylabel="Logistic Regression C",title="Ticket 4: Best Dev F1 per LR Configuration")
    for row_index in range(len(pivot.index)):
        for column_index in range(len(pivot.columns)):
            axis.text(column_index,row_index,f"{pivot.iloc[row_index,column_index]:.3f}",ha="center",va="center",fontsize=8)
    fig.colorbar(image,ax=axis,label="Dev F1 (target=1)")
    fig.tight_layout()
    fig.savefig(figures_path("ticket4_model_grid.png"),dpi=180)
    plt.close(fig)


def write_ticket5_audit_artifacts(audit):
    summary=(audit.assign(is_heldout=audit.split.eq("heldout"))
             .groupby(["issue_type","disposition"],as_index=False)
             .agg(evidence_rows=("id","size"),unique_ids=("id","nunique"),heldout_evidence_rows=("is_heldout","sum")))
    summary["heldout_evidence_rows"]=summary.heldout_evidence_rows.astype(int)
    write_csv(summary,RESULTS/"ticket5_audit_summary.csv",["issue_type","disposition"])
    pivot=summary.pivot(index="issue_type",columns="disposition",values="evidence_rows").fillna(0).sort_index()
    colors={"ambiguous":"#e67e22","keep_but_flag":"#15616d","reject_false_positive":"#c1121f"}
    fig,axis=plt.subplots(figsize=(7.2,4.5))
    left=np.zeros(len(pivot))
    for disposition in pivot.columns:
        values=pivot[disposition].to_numpy()
        axis.barh(pivot.index,values,left=left,label=disposition,color=colors.get(disposition,"#777777"))
        left+=values
    axis.set(xlabel="Audit evidence rows",title="Ticket 5: Audit Evidence by Type and Disposition")
    axis.legend(frameon=False,fontsize=8)
    axis.grid(axis="x",alpha=.2)
    fig.tight_layout()
    fig.savefig(figures_path("ticket5_audit_distribution.png"),dpi=180)
    plt.close(fig)


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


def _baseline_run(train, dev):
    builder, model = fit_model(train)
    scores, predictions = score(builder, model, dev)
    return builder, model, scores, predictions, binary_metrics(dev.target, predictions)


def _ticket4_dev_run(train, dev):
    candidates=[]
    for C in (.1,.25,.5,1,2,4,10):
        for weight in (None,"balanced"):
            builder,model=fit_model(train,C=C,class_weight=weight)
            scores,_=score(builder,model,dev)
            best=max((binary_metrics(dev.target,labels_from_scores(scores,threshold))["f1"],float(threshold))
                     for threshold in np.round(np.arange(.2,.801,.01),2))
            candidates.append((best[0],f"lr_c{C}_{weight or 'none'}_t{best[1]:.2f}","lr",C,weight,best[1],builder,model,scores,labels_from_scores(scores,best[1])))
    builder,model=fit_model(train,family="svc")
    scores,predictions=score(builder,model,dev,family="svc")
    candidates.append((binary_metrics(dev.target,predictions)["f1"],"linear_svc_c1","svc",1,None,0.0,builder,model,scores,predictions))
    return sorted(candidates,key=lambda row:(-row[0],row[1]))[0]


def run_ticket_dev(ticket_number: int) -> dict:
    """Select one ticket configuration using train/dev only and persist it as pending."""
    parts=get_train_dev(); train,dev=parts["train"],parts["dev"]
    ticket=f"ticket-{ticket_number}"
    baseline_builder,baseline_model,baseline_scores,baseline_predictions,baseline_metrics=_baseline_run(train,dev)
    if ticket_number==1:
        name="tfidf_12_sublinear_lr_liblinear_c1"; builder,model,scores,predictions=baseline_builder,baseline_model,baseline_scores,baseline_predictions
        record=decision(ticket,{"mode":"text","normalization":asdict(NormalizationConfig()),"vectorizer":builder.vectorizer.get_params(),"classifier":model.get_params()},.5,"dev_f1_target_1",baseline_metrics["f1"],"Reasonable text-only baseline fixed from the assignment and dev evaluation.","t1-baseline")
        record["diagnostic_probe_policy"]=TICKET1_PROBE_POLICY
    elif ticket_number==2:
        candidates=[]
        for name,cfg in NORMALIZATION_CANDIDATES.items():
            builder,model=fit_model(train,normalization=cfg); scores,predictions=score(builder,model,dev)
            candidates.append((binary_metrics(dev.target,predictions)["f1"],name,cfg,builder,model,scores,predictions))
        _,name,cfg,builder,model,scores,predictions=sorted(candidates,key=lambda row:(-row[0],row[1]))[0]
        record=decision(ticket,{"candidate":name,"normalization":asdict(cfg)},.5,"dev_f1_target_1",binary_metrics(dev.target,predictions)["f1"],"Highest dev F1 among controlled single-lever candidates; mechanism checked via transitions.","t2-normalization")
    elif ticket_number==3:
        candidates=[]
        for mode in ("text","keyword","location","numeric","keyword+location","text+keyword","text+numeric"):
            builder,model=fit_model(train,mode=mode); scores,predictions=score(builder,model,dev); name=mode.replace("+","_plus_")
            candidates.append((binary_metrics(dev.target,predictions)["f1"],name,mode,builder,model,scores,predictions))
        _,name,mode,builder,model,scores,predictions=sorted(candidates,key=lambda row:(-row[0],row[1]))[0]
        record=decision(ticket,{"mode":mode},.5,"dev_f1_target_1",binary_metrics(dev.target,predictions)["f1"],"Best controlled dev feature set; shortcut-only results retained for audit.","t3-features")
    else:
        _,name,family,C,weight,threshold,builder,model,scores,predictions=_ticket4_dev_run(train,dev)
        model_metrics=binary_metrics(dev.target,predictions)
        if ticket_number==4:
            record=decision(ticket,{"family":family,"C":C,"class_weight":weight,"score_type":"decision_function" if family=="svc" else "probability"},threshold,"dev_f1_target_1",model_metrics["f1"],"Best dev operating point in limited interpretable search.","t4-decision-rule")
        else:
            ticket="ticket-5"
            record=decision(ticket,{"label_correction":"none","audit_model":name},threshold,"dev_f1_target_1",model_metrics["f1"],"Audit evidence was insufficient for an automatic training-label correction.","t5-data-quality")
    pending=json.loads(PENDING_DECISIONS.read_text()) if PENDING_DECISIONS.exists() else {}
    pending[ticket]=record
    write_json(pending,PENDING_DECISIONS)
    write_csv(prediction_frame(dev,scores,predictions,name,ticket),PREDICTIONS/"dev"/f"{ticket}.csv",["id"])
    return record


def freeze_ticket(ticket_number: int) -> dict:
    ticket=f"ticket-{ticket_number}"
    if not PENDING_DECISIONS.exists():
        raise ValueError("No pending dev decision. Run the ticket dev experiment first.")
    pending=json.loads(PENDING_DECISIONS.read_text())
    if ticket not in pending:
        raise ValueError(f"No pending decision for {ticket}")
    record=pending.pop(ticket); record["status"]="frozen"
    frozen_path=EXPERIMENTS/"decisions.json"
    frozen=json.loads(frozen_path.read_text()) if frozen_path.exists() else {}
    frozen[ticket]=record
    write_json(pending,PENDING_DECISIONS)
    write_json(frozen,frozen_path)
    return record


def run_ticket_heldout(ticket_number: int) -> dict:
    """Score exactly one frozen configuration on held-out data."""
    ticket=f"ticket-{ticket_number}"; frozen_path=EXPERIMENTS/"decisions.json"
    if not frozen_path.exists(): raise ValueError("Held-out evaluation requires a frozen decision")
    frozen_all=json.loads(frozen_path.read_text()); frozen=frozen_all.get(ticket)
    if not frozen or frozen.get("status")!="frozen": raise ValueError("Held-out evaluation requires a frozen decision")
    parts=get_splits(); train,held=parts["train"],parts["heldout"]
    selected=frozen["selected_configuration"]
    if ticket_number==1:
        name="tfidf_12_sublinear_lr_liblinear_c1"; builder,model=fit_model(train); family="lr"
    elif ticket_number==2:
        name=selected["candidate"]; builder,model=fit_model(train,normalization=NormalizationConfig(**selected["normalization"])); family="lr"
    elif ticket_number==3:
        name=selected["mode"].replace("+","_plus_"); builder,model=fit_model(train,mode=selected["mode"]); family="lr"
    else:
        source=selected if ticket_number==4 else frozen_all.get("ticket-4",{}).get("selected_configuration")
        if not source: raise ValueError("Ticket 5 held-out evaluation requires frozen Ticket 4 model decision")
        name=selected.get("audit_model",f"lr_c{source['C']}_{source['class_weight'] or 'none'}_t{frozen['selected_threshold']:.2f}")
        family=source["family"]; builder,model=fit_model(train,C=source["C"],class_weight=source["class_weight"],family=family)
    scores,predictions=score(builder,model,held,frozen["selected_threshold"],family)
    frame=prediction_frame(held,scores,predictions,name,ticket)
    write_csv(frame,PREDICTIONS/"heldout"/f"{ticket}.csv",["id"])
    return {"ticket":ticket,"metrics":binary_metrics(held.target,predictions),"prediction_rows":len(frame)}


def run_all() -> dict:
    ensure_dirs(); split=get_splits(); train,dev,held=split["train"],split["dev"],split["heldout"]
    decisions=[]; registry=[]; summaries=[]; all_held=[]; all_trans=[]; confusions=[]; tops=[]; stress=[]; ablations=[]; diagnostic_transitions=[]
    decision_path=EXPERIMENTS/"decisions.json"
    write_json({},decision_path)
    def persist_decisions():
        for item in decisions: item["status"]="frozen"
        write_json({d["ticket"]:d for d in decisions},decision_path)

    # Ticket 1: assignment-driven baseline; freeze before first held-out scoring.
    bld,mdl=fit_model(train); ds,dp=score(bld,mdl,dev); dm=binary_metrics(dev.target,dp)
    bname="tfidf_12_sublinear_lr_liblinear_c1"; ticket="ticket-1"
    decisions.append(decision(ticket,{"mode":"text","normalization":asdict(NormalizationConfig()),"vectorizer":bld.vectorizer.get_params(),
        "classifier":mdl.get_params()},.5,"dev_f1_target_1",dm["f1"],"Reasonable text-only baseline fixed from the assignment and dev evaluation.","t1-baseline"))
    decisions[-1]["diagnostic_probe_policy"]=TICKET1_PROBE_POLICY
    persist_decisions()
    hs,hp=score(bld,mdl,held); hm=binary_metrics(held.target,hp)
    base_dev_pred,base_held_pred=dp.copy(),hp.copy(); base_dev_scores=ds.copy()
    write_csv(prediction_frame(dev,ds,dp,bname,ticket),PREDICTIONS/"dev"/"ticket-1.csv",["id"])
    all_held.append(prediction_frame(held,hs,hp,bname,ticket)); tops.append(top_coefficients(bld,mdl,ticket,bname))
    registry.append((ticket,bname,"baseline",dm["f1"],hm["f1"],.5))
    confusions.extend([(ticket,"dev",bname,*[dm[k] for k in ("tn","fp","fn","tp")]),(ticket,"heldout",bname,*[hm[k] for k in ("tn","fp","fn","tp")])])
    summaries.append((ticket,bname,dm["f1"],hm["f1"],hm["accuracy"],0,0,0,0,"accepted","Assignment-driven baseline frozen at threshold 0.5"))
    floor=np.zeros(len(dev),int); fm=binary_metrics(dev.target,floor); floor_h=binary_metrics(held.target,np.zeros(len(held),int))
    registry.append((ticket,"majority_floor","floor",fm["f1"],floor_h["f1"],.5))
    discrepancy=[("submitted_baseline","none","frozen configuration",SEED,len(bld.vectorizer.vocabulary_),
                  dm["precision"],dm["recall"],dm["f1"],hm["precision"],hm["recall"],hm["f1"],
                  0.0,0.0,hm["f1"]-.7574221578566256,int(mdl.n_iter_[0]))]
    diagnostic_probes=[
        ("unigrams_only","ngram_range","{'ngram_range': (1, 1)}",{"ngram_range":(1,1)},{}),
        ("trigrams_added","ngram_range","{'ngram_range': (1, 3)}",{"ngram_range":(1,3)},{}),
        ("no_sublinear_tf","sublinear_tf","{'sublinear_tf': False}",{"sublinear_tf":False},{}),
        ("max_features_20000","max_features","{'max_features': 20000}",{"max_features":20000},{}),
        ("max_features_unbounded","max_features","{'max_features': None}",{"max_features":None},{}),
        ("min_df_2","min_df","{'min_df': 2}",{"min_df":2},{}),
        ("min_df_3","min_df","{'min_df': 3}",{"min_df":3},{}),
        ("min_df_5","min_df","{'min_df': 5}",{"min_df":5},{}),
        ("max_df_0_9","max_df","{'max_df': 0.9}",{"max_df":.9},{}),
        ("no_idf","use_idf","{'use_idf': False}",{"use_idf":False},{}),
        ("no_smooth_idf","smooth_idf","{'smooth_idf': False}",{"smooth_idf":False},{}),
        ("l1_normalization","norm","{'norm': 'l1'}",{"norm":"l1"},{}),
        ("no_vector_normalization","norm","{'norm': None}",{"norm":None},{}),
        ("unicode_accents","strip_accents","{'strip_accents': 'unicode'}",{"strip_accents":"unicode"},{}),
        ("single_character_tokens","token_pattern","{'token_pattern': '(?u)\\b\\w+\\b'}",{"token_pattern":r"(?u)\b\w+\b"},{}),
        ("preserve_case","casing","{'casing': 'preserve'}",{}, {"normalization":NormalizationConfig(casing="preserve")}),
        ("c_0_1","C","{'C': 0.1}",{}, {"C":.1}),
        ("c_0_25","C","{'C': 0.25}",{}, {"C":.25}),
        ("c_0_5","C","{'C': 0.5}",{}, {"C":.5}),
        ("c_2","C","{'C': 2.0}",{}, {"C":2.0}),
        ("c_4","C","{'C': 4.0}",{}, {"C":4.0}),
        ("c_10","C","{'C': 10.0}",{}, {"C":10.0}),
        ("balanced_classes","class_weight","{'class_weight': 'balanced'}",{}, {"class_weight":"balanced"}),
        ("lbfgs_solver","solver","{'solver': 'lbfgs'}",{}, {"solver":"lbfgs"}),
        ("seed_1","random_state","{'random_state': 1}",{}, {"seed":1}),
        ("seed_42","random_state","{'random_state': 42}",{}, {"seed":42}),
    ]
    for label,factor,setting,vectorizer_kwargs,model_kwargs in diagnostic_probes:
        normalization=model_kwargs.pop("normalization",NormalizationConfig())
        pb,pm=fit_model(train,normalization=normalization,**vectorizer_kwargs,**model_kwargs)
        ps,pp=score(pb,pm,dev); phs,php=score(pb,pm,held); pdm=binary_metrics(dev.target,pp); phm=binary_metrics(held.target,php)
        registry.append((ticket,label,"diagnostic_probe",pdm["f1"],phm["f1"],.5))
        discrepancy.append((label,factor,setting,pm.random_state,len(pb.vectorizer.vocabulary_),pdm["precision"],pdm["recall"],
                pdm["f1"],phm["precision"],phm["recall"],phm["f1"],pdm["f1"]-dm["f1"],phm["f1"]-hm["f1"],phm["f1"]-.7574221578566256,
                            int(np.max(np.atleast_1d(pm.n_iter_)))))
        if label in TICKET1_TRANSITION_PROBES:
            transition=error_transitions(held.id,held.target,base_held_pred,php,bname,label)
            transition["probe"]=label
            diagnostic_transitions.append(transition)

    # Ticket 2: each candidate changes one normalization variable on dev.
    norm_runs=[]
    for name,cfg in NORMALIZATION_CANDIDATES.items():
        cb,cm=fit_model(train,normalization=cfg); cs,cp=score(cb,cm,dev); met=binary_metrics(dev.target,cp)
        trans=error_transitions(dev.id,dev.target,base_dev_pred,cp,bname,name); trans["ticket"]=ticket2="ticket-2"; all_trans.append(trans)
        norm_runs.append((met["f1"],name,cfg,cb,cm,cs,cp,met)); registry.append((ticket2,name,"single_normalization_lever",met["f1"],np.nan,.5))
    norm_runs.sort(key=lambda x:(-x[0],x[1])); _,nname,ncfg,nb,nm,nds,ndp,ndm=norm_runs[0]
    ticket2_candidates=pd.DataFrame([
        (name,metrics["precision"],metrics["recall"],metrics["f1"],metrics["f1"]-dm["f1"],name==nname)
        for _,name,_,_,_,_,_,metrics in norm_runs
    ],columns=["candidate","dev_precision_target_1","dev_recall_target_1","dev_f1_target_1","dev_delta_from_raw","selected"])
    ticket2_candidates["rank"]=ticket2_candidates.dev_f1_target_1.rank(ascending=False,method="min").astype(int)
    decisions.append(decision(ticket2,{"candidate":nname,"normalization":asdict(ncfg)},.5,"dev_f1_target_1",ndm["f1"],"Highest dev F1 among controlled single-lever candidates; mechanism checked via transitions.","t2-normalization"))
    persist_decisions()
    nhs,nhp=score(nb,nm,held); nhm=binary_metrics(held.target,nhp); nt=error_transitions(held.id,held.target,base_held_pred,nhp,bname,nname); nt["ticket"]=ticket2; all_trans.append(nt); nc=transition_counts(nt)
    write_csv(prediction_frame(dev,nds,ndp,nname,ticket2),PREDICTIONS/"dev"/"ticket-2.csv",["id"]); all_held.append(prediction_frame(held,nhs,nhp,nname,ticket2)); tops.append(top_coefficients(nb,nm,ticket2,nname))
    summaries.append((ticket2,nname,ndm["f1"],nhm["f1"],nhm["accuracy"],*[nc[k] for k in ("fixed_fp","fixed_fn","new_fp","new_fn")],"accepted",f"Dev-selected single lever: {nname}"))
    confusions.extend([(ticket2,"dev",nname,*[ndm[k] for k in ("tn","fp","fn","tp")]),(ticket2,"heldout",nname,*[nhm[k] for k in ("tn","fp","fn","tp")])])
    for kind in ("url_replace","mention_replace","lowercase","hashtag_strip","punct_remove","emoji_remove"):
        append_stress(stress,ticket2,bname,bld,mdl,dev,base_dev_pred,kind,"raw-model superficial-text sensitivity")
        append_stress(stress,ticket2,nname,nb,nm,dev,ndp,kind,"selected-normalization model sensitivity")

    # Ticket 3: controlled signal ablations/additions.
    modes=["text","keyword","location","numeric","keyword+location","text+keyword","text+numeric"]
    feat_runs=[]; feature_runs_by_name={}
    for mode in modes:
        fb,fm_=fit_model(train,mode=mode); fs,fp=score(fb,fm_,dev); met=binary_metrics(dev.target,fp); name=mode.replace("+","_plus_")
        feat_runs.append((met["f1"],name,mode,fb,fm_,fs,fp,met)); registry.append(("ticket-3",name,"feature_ablation",met["f1"],np.nan,.5))
        feature_runs_by_name[name]=(mode,fb,fm_,fp)
        tr=error_transitions(dev.id,dev.target,base_dev_pred,fp,bname,name); tr["ticket"]="ticket-3"; all_trans.append(tr)
        if hasattr(fm_,"coef_"): tops.append(top_coefficients(fb,fm_,"ticket-3",name,15))
    feat_runs.sort(key=lambda x:(-x[0],x[1])); _,fname,fmode,fb,fm_,fds,fdp,fdm=feat_runs[0]
    ticket3_candidates=pd.DataFrame([
        (mode,metrics["precision"],metrics["recall"],metrics["f1"],metrics["f1"]-dm["f1"],mode==fmode)
        for _,_,mode,_,_,_,_,metrics in feat_runs
    ],columns=["feature_mode","dev_precision_target_1","dev_recall_target_1","dev_f1_target_1","dev_delta_from_text","selected"])
    ticket3_candidates["rank"]=ticket3_candidates.dev_f1_target_1.rank(ascending=False,method="min").astype(int)
    decisions.append(decision("ticket-3",{"mode":fmode},.5,"dev_f1_target_1",fdm["f1"],"Best controlled dev feature set; shortcut-only results retained for audit.","t3-features"))
    persist_decisions()
    fhs,fhp=score(fb,fm_,held); fhm=binary_metrics(held.target,fhp); ft=error_transitions(held.id,held.target,base_held_pred,fhp,bname,fname); ft["ticket"]="ticket-3"; all_trans.append(ft); fc=transition_counts(ft)
    write_csv(prediction_frame(dev,fds,fdp,fname,"ticket-3"),PREDICTIONS/"dev"/"ticket-3.csv",["id"]); all_held.append(prediction_frame(held,fhs,fhp,fname,"ticket-3"))
    summaries.append(("ticket-3",fname,fdm["f1"],fhm["f1"],fhm["accuracy"],*[fc[k] for k in ("fixed_fp","fixed_fn","new_fp","new_fn")],"accepted",f"Dev-selected feature set: {fmode}"))
    confusions.extend([("ticket-3","dev",fname,*[fdm[k] for k in ("tn","fp","fn","tp")]),("ticket-3","heldout",fname,*[fhm[k] for k in ("tn","fp","fn","tp")])])
    metadata_stress={
        "keyword":("keyword_blank","keyword_shuffle"),
        "location":("location_blank","location_shuffle"),
        "keyword_plus_location":("keyword_blank","keyword_shuffle","location_blank","location_shuffle"),
        "text_plus_keyword":("keyword_blank","keyword_shuffle"),
    }
    for name,kinds in metadata_stress.items():
        _,stress_builder,stress_model,stress_pred=feature_runs_by_name[name]
        for kind in kinds:
            append_stress(stress,"ticket-3",name,stress_builder,stress_model,dev,stress_pred,kind,
                          "metadata shortcut sensitivity on a model that consumes this field")

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
    ticket4_candidates=pd.DataFrame([
        (name,candidate_family,candidate_C,candidate_weight or "none",candidate_threshold,metrics["precision"],metrics["recall"],metrics["f1"],name==mname)
        for _,name,candidate_family,candidate_C,candidate_weight,candidate_threshold,_,_,_,_,metrics in candidates
    ],columns=["model_name","family","C","class_weight","threshold","dev_precision_target_1","dev_recall_target_1","dev_f1_target_1","selected"])
    ablation_configs=[
        ("baseline_fixed_t0.50",bld,mdl,ds,dp,.5,"Ticket 1 baseline"),
        ("baseline_threshold_t0.46",bld,mdl,ds,labels_from_scores(ds,.46),.46,"threshold only"),
    ]
    for name,chosen_family,chosen_c,chosen_weight,chosen_threshold in [
        ("lr_c10_none_t0.52","lr",10,None,.52),
        ("lr_c1_balanced_t0.51","lr",1,"balanced",.51),
        (mname,family,C,weight,threshold),
    ]:
        candidate=next(item for item in candidates if item[1]==name)
        _,_,_,_,_,candidate_threshold,candidate_builder,candidate_model,candidate_scores,candidate_pred,_=candidate
        ablation_configs.append((name,candidate_builder,candidate_model,candidate_scores,candidate_pred,candidate_threshold,
                                  "selected combination" if name==mname else "single model-configuration change from baseline"))
    for name,_,_,scores,prediction,selected_threshold,description in ablation_configs:
        metrics=binary_metrics(dev.target,prediction)
        ablations.append(("ticket-4",name,description,selected_threshold,metrics["precision"],metrics["recall"],metrics["f1"],
                          metrics["tn"],metrics["fp"],metrics["fn"],metrics["tp"]))
    decisions.append(decision("ticket-4",{"family":family,"C":C,"class_weight":weight,"score_type":"decision_function" if family=="svc" else "probability"},threshold,"dev_f1_target_1",mdm["f1"],"Best dev operating point in limited interpretable search.","t4-decision-rule"))
    if family=="lr":
        write_ticket4_decision_curves(dev.target,mds,threshold)
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
    # These are manually reviewed non-events, not label corrections.
    rejected_model_findings={198:"lifetime-risk discussion is not a reported disaster event"}
    # High-confidence held-out errors: analysis, never automatic relabels.
    for pos,row in held.reset_index(drop=True).iterrows():
        if mhp[pos]!=row.target and (mhs[pos]>.8 or mhs[pos]<.2):
            typ="high_confidence_fp" if mhp[pos]==1 else "high_confidence_fn"
            finding=rejected_model_findings.get(int(row.id))
            disposition="reject_false_positive" if finding else "ambiguous"
            rationale=f"manual semantic review rejected model finding: {finding}" if finding else "model disagreement is not label proof"
            snippet=" ".join(str(row.text)[:140].split())
            audit.append((int(row.id),typ,f"model_score={mhs[pos]:.4f}; text={snippet}",disposition,float(max(mhs[pos],1-mhs[pos])),"heldout",int(row.target),"","",rationale))
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
    discrepancy_df=pd.DataFrame(discrepancy,columns=["probe","single_factor","setting","random_state","vocabulary_size",
                                                "dev_precision_target_1","dev_recall_target_1","dev_f1_target_1",
                                                "heldout_precision_target_1","heldout_recall_target_1","heldout_f1_target_1",
                                                "dev_delta_from_baseline","heldout_delta_from_baseline","difference_from_reference","iterations"])
    write_csv(discrepancy_df,RESULTS/"discrepancy_comparison.csv",["probe"])
    diagnostic_deltas=discrepancy_df[discrepancy_df.probe!="submitted_baseline"]
    pearson=pearsonr(diagnostic_deltas.dev_delta_from_baseline,diagnostic_deltas.heldout_delta_from_baseline)
    spearman=spearmanr(diagnostic_deltas.dev_delta_from_baseline,diagnostic_deltas.heldout_delta_from_baseline)
    write_json({"baseline_probe":"submitted_baseline","probe_count":int(len(diagnostic_deltas)),
                "dev_metric":"f1_target_1","heldout_metric":"f1_target_1",
                "pearson_r":float(pearson.statistic),"pearson_p_value":float(pearson.pvalue),
                "spearman_rho":float(spearman.statistic),"spearman_p_value":float(spearman.pvalue),
                "policy":TICKET1_PROBE_POLICY},RESULTS/"discrepancy_association.json")
    diagnostic_transition_df=pd.concat(diagnostic_transitions)
    write_csv(diagnostic_transition_df,RESULTS/"discrepancy_error_transitions.csv",["probe","id"])
    transition_summary=diagnostic_transition_df.pivot_table(index="probe",columns="category",aggfunc="size",fill_value=0)
    transition_summary=transition_summary.reindex(columns=["fixed_fp","fixed_fn","new_fp","new_fn"],fill_value=0).reset_index()
    write_csv(transition_summary,RESULTS/"discrepancy_transition_summary.csv",["probe"])
    write_csv(pd.concat(all_held),PREDICTIONS/"heldout_predictions.csv",["ticket","id"])
    write_csv(pd.DataFrame(summaries,columns=["ticket","model_name","dev_f1_target_1","heldout_f1_target_1","heldout_accuracy","fixed_fp","fixed_fn","new_fp","new_fn","decision","decision_reason"]),RESULTS/"summary.csv",["ticket"])
    write_csv(pd.DataFrame(sweep,columns=["ticket","threshold","precision_target_1","recall_target_1","f1_target_1","tp","fp","tn","fn","predicted_positive_count"]),RESULTS/"threshold_sweep.csv",["threshold"])
    write_csv(pd.DataFrame(ablations,columns=["ticket","model_name","comparison","threshold","precision_target_1","recall_target_1","f1_target_1","tn","fp","fn","tp"]),RESULTS/"decision_ablation.csv",["ticket","model_name"])
    write_csv(adf,RESULTS/"data_quality_audit.csv",["id","issue_type"])
    write_csv(pd.concat(all_trans),RESULTS/"error_transitions.csv",["ticket","id","candidate_model"])
    stress_df=pd.DataFrame(stress,columns=["ticket","model_name","perturbation","baseline_metric","perturbed_metric","absolute_change","relative_change","prediction_flips","fixed_fp","fixed_fn","new_fp","new_fn","interpretation"])
    write_csv(stress_df,RESULTS/"perturbation_stress.csv",["ticket","model_name","perturbation"])
    write_csv(pd.DataFrame(confusions,columns=["ticket","split","model_name","tn","fp","fn","tp"]),RESULTS/"confusion_matrices.csv",["ticket","split"])
    write_csv(pd.concat(tops),RESULTS/"top_features.csv",["ticket","model_name","direction","coefficient"])
    write_ticket1_probe_agreement_figure(discrepancy_df)
    write_ticket2_evidence_artifacts(ticket2_candidates,stress_df)
    write_ticket3_evidence_artifacts(ticket3_candidates,stress_df)
    write_ticket4_grid_artifacts(ticket4_candidates)
    write_ticket5_audit_artifacts(adf)
    write_json({"python":platform.python_version(),"scikit_learn":sklearn.__version__,"pandas":pd.__version__,"numpy":np.__version__,"scipy":scipy.__version__,"seed":SEED,"cpu_only":True},RESULTS/"environment.json")
    summary_columns=["ticket","model_name","dev_f1_target_1","heldout_f1_target_1","heldout_accuracy","fixed_fp","fixed_fn","new_fp","new_fn","decision","decision_reason"]
    return {"summary":pd.DataFrame(summaries,columns=summary_columns),"floor":fm,"audit_rows":len(adf),"decisions":decisions}
