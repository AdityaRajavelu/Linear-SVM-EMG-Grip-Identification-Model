"""Train a session-aware linear SVM on 4-channel EMG data to classify gestures.

Training is cumulative: it uses every recording session listed in
emg_data.SESSIONS (day1 = original files, day2 = *_day2.txt files, ...).
Feature engineering is in emg_data.EMGModel.

Data files per session (one row per sample, all aligned):
  emg_output*.txt             - 4 comma-separated EMG channel values
  relabel_output*.txt         - gesture label (1 = rest, 2-6 = gestures); cleaned labels
  gestureCounter_output*.txt  - repetition index, used to split train/test
                                so that whole repetitions are held out (no leakage
                                between temporally adjacent samples).
"""
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, GroupKFold

from emg_data import SESSIONS, TEST_REP_START, EMGModel, load_all

X, y, reps, session, group = load_all()

# Hold out the last repetitions (>= TEST_REP_START) of every session as the test set.
test_mask = reps >= TEST_REP_START
train_mask = ~test_mask
print(f"sessions: {list(SESSIONS)}")
print(f"train: {train_mask.sum()} samples, test: {test_mask.sum()} samples")

model = EMGModel(class_weight="balanced")
# Normalization statistics come from training repetitions only.
model.fit_normalization(X[train_mask], session[train_mask])
# Features are computed on the full, continuous recordings (temporal windows are
# causal, so a test sample only ever sees past samples), then split.
F = model.transform(X, session)
print(f"feature dimension: {F.shape[1]}")

# Tune C with grouped CV: groups are (session, repetition) pairs.
search = GridSearchCV(
    model.make_classifier(),
    {"linearsvc__C": [0.1, 1, 10]},
    cv=GroupKFold(n_splits=4),
    scoring="balanced_accuracy",
    n_jobs=-1,
    return_train_score=False,
)
search.fit(F[train_mask], y[train_mask], groups=group[train_mask])
print("best params:", search.best_params_)
print(f"CV balanced accuracy: {search.best_score_:.4f}")

model.pipeline_ = search.best_estimator_
model.C = search.best_params_["linearsvc__C"]
model.classes_ = model.pipeline_.classes_
model.cv_results_ = {
    k: v for k, v in search.cv_results_.items()
    if k.startswith(("param_", "mean_test", "std_test", "split"))
}

y_pred = model.predict_features(F[test_mask])
y_test, s_test = y[test_mask], session[test_mask]
print(f"\nTest accuracy (all sessions): {accuracy_score(y_test, y_pred):.4f}")
for name in SESSIONS:
    m = s_test == name
    print(f"  {name}: {accuracy_score(y_test[m], y_pred[m]):.4f}")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "linear_svm.joblib")
print("\nSaved model to linear_svm.joblib")
