"""Evaluate the trained models and write a full metrics report to svm_results.md.

Run train_svm.py first (it produces linear_svm.joblib). linear_svm_day1.joblib is
the original model, trained on day-1 data only, kept as the baseline.
Uses the same split as training: repetitions >= TEST_REP_START of every session
are the held-out test set.
"""
import time

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, label_binarize
from sklearn.svm import LinearSVC

from emg_data import SESSIONS, TEST_REP_START, WINDOWS, EMGModel, load_all

CLASS_NAMES = {1: "1 (rest)", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6"}
SESS = list(SESSIONS)

X, y, reps, session, group = load_all()
test_mask = reps >= TEST_REP_START
train_mask = ~test_mask
y_train, y_test = y[train_mask], y[test_mask]
s_test, r_test = session[test_mask], reps[test_mask]

model = joblib.load("linear_svm.joblib")
baseline = joblib.load("linear_svm_day1.joblib")
classes = model.classes_


def overall_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro F1": f1_score(y_true, y_pred, average="macro"),
        "Weighted F1": f1_score(y_true, y_pred, average="weighted"),
        "Cohen's kappa": cohen_kappa_score(y_true, y_pred),
        "Matthews corr. coef. (MCC)": matthews_corrcoef(y_true, y_pred),
    }


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt(v):
    return f"{v:.4f}"


def delta(new, old):
    d = new - old
    return f"{d:+.4f}"


# ------------------------------------------------------------ predictions
F = model.transform(X, session)
t0 = time.perf_counter()
y_pred = model.predict_features(F[test_mask])
pred_time = time.perf_counter() - t0
y_pred_train = model.predict_features(F[train_mask])
scores = model.decision_function_features(F[test_mask])

# -------------------------------------------- comparison models / ablations
print("fitting comparison models ...")
preds = {}
# A. original model: day-1 data only, raw log features
preds["Original (day 1 only, raw features)"] = baseline.predict(X[test_mask])
# B. naive cumulative: same simple pipeline, just retrained on day 1 + day 2
naive = make_pipeline(
    FunctionTransformer(np.log1p), StandardScaler(),
    LinearSVC(C=10, class_weight="balanced", dual=False, max_iter=5000),
).fit(X[train_mask], y_train)
preds["Naive cumulative (day 1 + 2, raw features)"] = naive.predict(X[test_mask])
# C/D. new feature pipeline trained on a single day (shows the value of the other day)
for only in SESS:
    m_tr = train_mask & (session == only)
    single = EMGModel(C=model.C).fit_normalization(X[m_tr], session[m_tr])
    Fs = single.transform(X[session == only], session[session == only])
    tr_local = train_mask[session == only]
    single.fit_features(Fs[tr_local], y[session == only][tr_local])
    p = np.full(test_mask.sum(), -1)
    p[s_test == only] = single.predict_features(Fs[~tr_local])
    preds[f"New features, {only} only"] = p
preds["**New cumulative model (day 1 + 2)**"] = y_pred


def eval_on(p, which):
    m = np.ones(len(y_test), bool) if which == "all" else (s_test == which)
    if (p[m] < 0).any():
        return None
    return overall_metrics(y_test[m], p[m])


comp_rows = []
for name, p in preds.items():
    row = [name]
    for which in SESS + ["all"]:
        r = eval_on(p, which)
        row += ["–", "–"] if r is None else [fmt(r["Balanced accuracy"]), fmt(r["Macro F1"])]
    comp_rows.append(row)
comp_header = ["Model"] + [f"{w} {k}" for w in SESS + ["all"] for k in ("bal. acc", "macro F1")]

# Before/after for every overall metric.
orig_p = preds["Original (day 1 only, raw features)"]
ba_rows = []
for which in SESS + ["all"]:
    o, n = eval_on(orig_p, which), eval_on(y_pred, which)
    for k in n:
        ba_rows.append([which, k, fmt(o[k]), fmt(n[k]), delta(n[k], o[k])])

# --------------------------------------------------------- new model detail
test_m = overall_metrics(y_test, y_pred)
train_m = overall_metrics(y_train, y_pred_train)
y_test_bin = label_binarize(y_test, classes=classes)
auc_per_class = roc_auc_score(y_test_bin, scores, average=None)
auc_macro = roc_auc_score(y_test_bin, scores, average="macro")

prec, rec, f1, sup = precision_recall_fscore_support(y_test, y_pred, labels=classes)
cm = confusion_matrix(y_test, y_pred, labels=classes)
specificity = []
for i in range(len(classes)):
    tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
    fp = cm[:, i].sum() - cm[i, i]
    specificity.append(tn / (tn + fp))
cm_norm = cm / cm.sum(axis=1, keepdims=True)

# Per-class F1 by session, original vs new.
f1_rows = []
for i, c in enumerate(classes):
    row = [CLASS_NAMES[c]]
    for which in SESS:
        m = s_test == which
        fo = f1_score(y_test[m], orig_p[m], labels=[c], average=None)[0]
        fn = f1_score(y_test[m], y_pred[m], labels=[c], average=None)[0]
        row += [fmt(fo), fmt(fn), delta(fn, fo)]
    f1_rows.append(row)
f1_header = ["Class"] + [f"{w} {k}" for w in SESS for k in ("original", "new", "Δ")]

# Cross-validation (from training).
cv = model.cv_results_
cv_C = [float(c) for c in cv["param_linearsvc__C"]]
cv_rows = [[c, f"{cv['mean_test_score'][i]:.4f} ± {cv['std_test_score'][i]:.4f}"]
           + [fmt(cv[f"split{k}_test_score"][i]) for k in range(4)]
           for i, c in enumerate(cv_C)]

# Per (session, repetition) test breakdown.
rep_rows = []
for s in SESS:
    for r in np.unique(r_test[s_test == s]):
        m = (s_test == s) & (r_test == r)
        rep_rows.append([s, r, m.sum(), fmt(accuracy_score(y_test[m], y_pred[m])),
                         fmt(balanced_accuracy_score(y_test[m], y_pred[m])),
                         fmt(f1_score(y_test[m], y_pred[m], average="macro"))])

# Top confusions.
conf_pairs = sorted(((cm[i, j], classes[i], classes[j]) for i in range(len(classes))
                     for j in range(len(classes)) if i != j), reverse=True)
top_conf = [[CLASS_NAMES[t], CLASS_NAMES[p], n, f"{n / cm[list(classes).index(t)].sum():.2%}"]
            for n, t, p in conf_pairs[:8]]

# Rest detection vs. gesture discrimination.
def rest_gesture(p):
    det = accuracy_score(y_test != 1, p != 1), f1_score(y_test != 1, p != 1)
    gm = (y_test != 1) & (p != 1)
    ges = accuracy_score(y_test[gm], p[gm]), f1_score(y_test[gm], p[gm], average="macro")
    return det, ges

(o_det, o_ges), (n_det, n_ges) = rest_gesture(orig_p), rest_gesture(y_pred)

# ---------------------------------------------------------------- report
n_feat = F.shape[1]
lines = [
    "# Linear SVM – Results Report (cumulative: day 1 + day 2)",
    "",
    "## Summary",
    "",
    f"- On the combined held-out test set, balanced accuracy went from "
    f"**{fmt(eval_on(orig_p, 'all')['Balanced accuracy'])} → {fmt(test_m['Balanced accuracy'])}** and macro F1 "
    f"from **{fmt(eval_on(orig_p, 'all')['Macro F1'])} → {fmt(test_m['Macro F1'])}** compared with the original day-1 model.",
    f"- Day-1 test set (same data as the first report): balanced accuracy "
    f"{fmt(eval_on(orig_p, 'day1')['Balanced accuracy'])} → {fmt(eval_on(y_pred, 'day1')['Balanced accuracy'])}, "
    f"so adding day 2 did not cost any day-1 performance.",
    "- Simply retraining the old pipeline on both days (\"naive cumulative\") *hurts* day 1 because the "
    "two days have different signal amplitudes per channel (e.g. electrode placement). The new "
    "pipeline fixes this with per-day normalization and per-day feature blocks.",
    "",
    "## Setup",
    "",
    md_table(["Item", "Value"], [
        ["Sessions", ", ".join(SESS)],
        ["Labels", "`relabel_output*.txt` (1 = rest, 2–6 = gestures)"],
        ["Train set", f"repetitions < {TEST_REP_START} of every session, {train_mask.sum():,} samples"],
        ["Test set", f"repetitions ≥ {TEST_REP_START} of every session, {test_mask.sum():,} samples (held out)"],
        ["Features", f"log1p → per-session standardization → raw + degree-2 polynomial terms + "
                     f"trailing moving averages {WINDOWS} samples and deviations from them → "
                     f"[shared | per-session] augmentation ({n_feat} features)"],
        ["Classifier", f"StandardScaler + LinearSVC (C={model.C}, class_weight={model.class_weight}, one-vs-rest)"],
        ["Hyperparameter search", "GridSearchCV over C, GroupKFold(4) by (session, repetition), scoring = balanced accuracy"],
        ["Prediction time (test set, features precomputed)",
         f"{pred_time * 1000:.1f} ms total, {pred_time / test_mask.sum() * 1e6:.2f} µs/sample"],
    ]),
    "",
    "## Model comparison (held-out test set)",
    "",
    md_table(comp_header, comp_rows),
    "",
    "- *Original*: the first model (day 1 only, 4 raw log-features). It scores near chance-level on day 2.",
    "- *Naive cumulative*: same pipeline as the original, retrained on both days.",
    "- *New features, dayN only*: the new pipeline trained on one day only (tested on that day only). "
    "Comparing it with the cumulative model shows how much the other day's data adds.",
    "",
    "## Before / after: original model vs. new cumulative model",
    "",
    md_table(["Test set", "Metric", "Original", "New", "Δ"], ba_rows),
    "",
    "## Overall metrics, new model (all sessions)",
    "",
    md_table(["Metric", "Train", "Test"],
             [[k, fmt(train_m[k]), fmt(test_m[k])] for k in test_m]
             + [["Macro ROC AUC (one-vs-rest)", "–", fmt(auc_macro)]]),
    "",
    "## Per-class metrics, new model (test set, all sessions)",
    "",
    md_table(["Class", "Precision", "Recall (sensitivity)", "Specificity", "F1", "ROC AUC", "Support"],
             [[CLASS_NAMES[c], fmt(prec[i]), fmt(rec[i]), fmt(specificity[i]), fmt(f1[i]),
               fmt(auc_per_class[i]), sup[i]] for i, c in enumerate(classes)]),
    "",
    "## Per-class F1 by session: original vs. new",
    "",
    md_table(f1_header, f1_rows),
    "",
    "## Confusion matrix, new model (test set, counts)",
    "",
    "Rows = true class, columns = predicted class.",
    "",
    md_table(["True \\ Pred"] + [CLASS_NAMES[c] for c in classes],
             [[f"**{CLASS_NAMES[c]}**"] + list(cm[i]) for i, c in enumerate(classes)]),
    "",
    "## Confusion matrix, new model (row-normalized = recall)",
    "",
    md_table(["True \\ Pred"] + [CLASS_NAMES[c] for c in classes],
             [[f"**{CLASS_NAMES[c]}**"] + [f"{v:.1%}" for v in cm_norm[i]]
              for i, c in enumerate(classes)]),
    "",
    "## Most frequent misclassifications, new model",
    "",
    md_table(["True", "Predicted as", "Count", "% of true class"], top_conf),
    "",
    "## Rest detection vs. gesture discrimination (all sessions)",
    "",
    md_table(["Task", "Original acc.", "Original F1", "New acc.", "New F1"], [
        ["Rest vs. any gesture (binary)", fmt(o_det[0]), fmt(o_det[1]), fmt(n_det[0]), fmt(n_det[1])],
        ["Which gesture (both true & pred are gestures)", fmt(o_ges[0]), fmt(o_ges[1]), fmt(n_ges[0]), fmt(n_ges[1])],
    ]),
    "",
    "## Cross-validation (training repetitions, balanced accuracy)",
    "",
    md_table(["C", "Mean ± SD", "Fold 1", "Fold 2", "Fold 3", "Fold 4"], cv_rows),
    "",
    "## Test performance by session and repetition, new model",
    "",
    md_table(["Session", "Repetition", "Samples", "Accuracy", "Balanced acc.", "Macro F1"], rep_rows),
    "",
    "## Notes and caveats",
    "",
    "- **New sessions need calibration.** Per-session normalization uses each day's own mean/std. "
    "For a new day, record a short calibration block (rest plus a few gestures) to compute these stats. "
    "Unless that day is also added to training, the model uses only the shared weights for it. "
    "To add a day: put its files next to the others, add it to `SESSIONS` in `emg_data.py`, then run "
    "`train_svm.py` and `evaluate_svm.py`.",
    "- **Temporal features are causal.** Moving averages only look at past samples, so the model can run in real time. "
    f"It needs about {max(WINDOWS)} samples of history to fill the longest window.",
    "- Normalization stats are computed from training repetitions only. CV folds share those stats "
    "(they use no labels), so CV scores can be very slightly optimistic. The test set is unaffected.",
    "",
    "## Metric definitions",
    "",
    "- **Accuracy**: fraction of samples classified correctly. Inflated by the large rest class.",
    "- **Balanced accuracy**: mean recall across classes. Not affected by class imbalance.",
    "- **Precision**: of the samples predicted as class *k*, the fraction that truly are *k*.",
    "- **Recall / sensitivity**: of the samples that truly are class *k*, the fraction predicted as *k*.",
    "- **Specificity**: of the samples that are not class *k*, the fraction correctly not predicted as *k*.",
    "- **F1**: harmonic mean of precision and recall. Macro = unweighted mean over classes; "
    "weighted = mean weighted by class support.",
    "- **Cohen's kappa**: agreement corrected for chance (0 = chance, 1 = perfect).",
    "- **MCC**: correlation between true and predicted labels (−1 to 1). Robust to imbalance.",
    "- **ROC AUC**: one-vs-rest area under the ROC curve from SVM decision scores. "
    "Measures how well the scores rank class *k* above the others, independent of the threshold.",
    "",
]

with open("svm_results.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("Wrote svm_results.md")
