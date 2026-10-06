# Linear SVM – Results Report (cumulative: day 1 + day 2)

## Summary

- On the combined held-out test set, balanced accuracy went from **0.7004 → 0.9449** and macro F1 from **0.6923 → 0.9422** compared with the original day-1 model.
- Day-1 test set (same data as the first report): balanced accuracy 0.9078 → 0.9379, so adding day 2 did not cost any day-1 performance.
- Simply retraining the old pipeline on both days ("naive cumulative") *hurts* day 1 because the two days have different signal amplitudes per channel (e.g. electrode placement). The new pipeline fixes this with per-day normalization and per-day feature blocks.

## Setup

| Item | Value |
|---|---|
| Sessions | day1, day2 |
| Labels | `relabel_output*.txt` (1 = rest, 2–6 = gestures) |
| Train set | repetitions < 8 of every session, 216,375 samples |
| Test set | repetitions ≥ 8 of every session, 99,421 samples (held out) |
| Features | log1p → per-session standardization → raw + degree-2 polynomial terms + trailing moving averages (25, 100, 400) samples and deviations from them → [shared | per-session] augmentation (126 features) |
| Classifier | StandardScaler + LinearSVC (C=0.1, class_weight=balanced, one-vs-rest) |
| Hyperparameter search | GridSearchCV over C, GroupKFold(4) by (session, repetition), scoring = balanced accuracy |
| Prediction time (test set, features precomputed) | 193.8 ms total, 1.95 µs/sample |

## Model comparison (held-out test set)

| Model | day1 bal. acc | day1 macro F1 | day2 bal. acc | day2 macro F1 | all bal. acc | all macro F1 |
|---|---|---|---|---|---|---|
| Original (day 1 only, raw features) | 0.9078 | 0.8963 | 0.4929 | 0.4321 | 0.7004 | 0.6923 |
| Naive cumulative (day 1 + 2, raw features) | 0.7083 | 0.6815 | 0.8413 | 0.8343 | 0.7780 | 0.7576 |
| New features, day1 only | 0.9399 | 0.9309 | – | – | – | – |
| New features, day2 only | – | – | 0.9524 | 0.9531 | – | – |
| **New cumulative model (day 1 + 2)** | 0.9379 | 0.9292 | 0.9535 | 0.9547 | 0.9449 | 0.9422 |

- *Original*: the first model (day 1 only, 4 raw log-features). It scores near chance-level on day 2.
- *Naive cumulative*: same pipeline as the original, retrained on both days.
- *New features, dayN only*: the new pipeline trained on one day only (tested on that day only). Comparing it with the cumulative model shows how much the other day's data adds.

## Before / after: original model vs. new cumulative model

| Test set | Metric | Original | New | Δ |
|---|---|---|---|---|
| day1 | Accuracy | 0.9370 | 0.9559 | +0.0189 |
| day1 | Balanced accuracy | 0.9078 | 0.9379 | +0.0301 |
| day1 | Macro F1 | 0.8963 | 0.9292 | +0.0329 |
| day1 | Weighted F1 | 0.9377 | 0.9564 | +0.0187 |
| day1 | Cohen's kappa | 0.9047 | 0.9330 | +0.0283 |
| day1 | Matthews corr. coef. (MCC) | 0.9053 | 0.9339 | +0.0286 |
| day2 | Accuracy | 0.7318 | 0.9703 | +0.2385 |
| day2 | Balanced accuracy | 0.4929 | 0.9535 | +0.4606 |
| day2 | Macro F1 | 0.4321 | 0.9547 | +0.5226 |
| day2 | Weighted F1 | 0.6878 | 0.9701 | +0.2823 |
| day2 | Cohen's kappa | 0.5919 | 0.9553 | +0.3634 |
| day2 | Matthews corr. coef. (MCC) | 0.6031 | 0.9554 | +0.3523 |
| all | Accuracy | 0.8329 | 0.9632 | +0.1303 |
| all | Balanced accuracy | 0.7004 | 0.9449 | +0.2445 |
| all | Macro F1 | 0.6923 | 0.9422 | +0.2499 |
| all | Weighted F1 | 0.8270 | 0.9631 | +0.1362 |
| all | Cohen's kappa | 0.7468 | 0.9444 | +0.1976 |
| all | Matthews corr. coef. (MCC) | 0.7510 | 0.9447 | +0.1938 |

## Overall metrics, new model (all sessions)

| Metric | Train | Test |
|---|---|---|
| Accuracy | 0.9832 | 0.9632 |
| Balanced accuracy | 0.9843 | 0.9449 |
| Macro F1 | 0.9785 | 0.9422 |
| Weighted F1 | 0.9833 | 0.9631 |
| Cohen's kappa | 0.9754 | 0.9444 |
| Matthews corr. coef. (MCC) | 0.9755 | 0.9447 |
| Macro ROC AUC (one-vs-rest) | – | 0.9961 |

## Per-class metrics, new model (test set, all sessions)

| Class | Precision | Recall (sensitivity) | Specificity | F1 | ROC AUC | Support |
|---|---|---|---|---|---|---|
| 1 (rest) | 0.9918 | 0.9827 | 0.9901 | 0.9872 | 0.9992 | 54446 |
| 2 | 0.9856 | 0.9991 | 0.9986 | 0.9923 | 1.0000 | 8728 |
| 3 | 0.8738 | 0.9876 | 0.9834 | 0.9272 | 0.9986 | 10377 |
| 4 | 0.9784 | 0.8060 | 0.9983 | 0.8839 | 0.9918 | 8556 |
| 5 | 0.8605 | 0.9457 | 0.9866 | 0.9011 | 0.9898 | 8013 |
| 6 | 0.9750 | 0.9482 | 0.9975 | 0.9614 | 0.9973 | 9301 |

## Per-class F1 by session: original vs. new

| Class | day1 original | day1 new | day1 Δ | day2 original | day2 new | day2 Δ |
|---|---|---|---|---|---|---|
| 1 (rest) | 0.9844 | 0.9892 | +0.0048 | 0.9752 | 0.9852 | +0.0100 |
| 2 | 0.9889 | 0.9942 | +0.0054 | 0.9583 | 0.9905 | +0.0322 |
| 3 | 0.8686 | 0.8608 | -0.0078 | 0.6535 | 0.9862 | +0.3328 |
| 4 | 0.8210 | 0.8696 | +0.0486 | 0.0000 | 0.8993 | +0.8993 |
| 5 | 0.8201 | 0.9311 | +0.1110 | 0.0056 | 0.8742 | +0.8686 |
| 6 | 0.8948 | 0.9299 | +0.0352 | 0.0000 | 0.9929 | +0.9929 |

## Confusion matrix, new model (test set, counts)

Rows = true class, columns = predicted class.

| True \ Pred | 1 (rest) | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| **1 (rest)** | 53503 | 127 | 240 | 66 | 404 | 106 |
| **2** | 8 | 8720 | 0 | 0 | 0 | 0 |
| **3** | 9 | 0 | 10248 | 0 | 0 | 120 |
| **4** | 53 | 0 | 783 | 6896 | 824 | 0 |
| **5** | 349 | 0 | 0 | 86 | 7578 | 0 |
| **6** | 25 | 0 | 457 | 0 | 0 | 8819 |

## Confusion matrix, new model (row-normalized = recall)

| True \ Pred | 1 (rest) | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| **1 (rest)** | 98.3% | 0.2% | 0.4% | 0.1% | 0.7% | 0.2% |
| **2** | 0.1% | 99.9% | 0.0% | 0.0% | 0.0% | 0.0% |
| **3** | 0.1% | 0.0% | 98.8% | 0.0% | 0.0% | 1.2% |
| **4** | 0.6% | 0.0% | 9.2% | 80.6% | 9.6% | 0.0% |
| **5** | 4.4% | 0.0% | 0.0% | 1.1% | 94.6% | 0.0% |
| **6** | 0.3% | 0.0% | 4.9% | 0.0% | 0.0% | 94.8% |

## Most frequent misclassifications, new model

| True | Predicted as | Count | % of true class |
|---|---|---|---|
| 4 | 5 | 824 | 9.63% |
| 4 | 3 | 783 | 9.15% |
| 6 | 3 | 457 | 4.91% |
| 1 (rest) | 5 | 404 | 0.74% |
| 5 | 1 (rest) | 349 | 4.36% |
| 1 (rest) | 3 | 240 | 0.44% |
| 1 (rest) | 2 | 127 | 0.23% |
| 3 | 6 | 120 | 1.16% |

## Rest detection vs. gesture discrimination (all sessions)

| Task | Original acc. | Original F1 | New acc. | New F1 |
|---|---|---|---|---|
| Rest vs. any gesture (binary) | 0.9779 | 0.9757 | 0.9860 | 0.9847 |
| Which gesture (both true & pred are gestures) | 0.6731 | 0.6486 | 0.9490 | 0.9480 |

## Cross-validation (training repetitions, balanced accuracy)

| C | Mean ± SD | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| 0.1 | 0.8840 ± 0.0588 | 0.8029 | 0.9634 | 0.8631 | 0.9064 |
| 1.0 | 0.8715 ± 0.0743 | 0.7732 | 0.9631 | 0.8301 | 0.9195 |
| 10.0 | 0.8741 ± 0.0694 | 0.7899 | 0.9587 | 0.8237 | 0.9240 |

## Test performance by session and repetition, new model

| Session | Repetition | Samples | Accuracy | Balanced acc. | Macro F1 |
|---|---|---|---|---|---|
| day1 | 8 | 14886 | 0.9609 | 0.9514 | 0.9399 |
| day1 | 9 | 14547 | 0.9436 | 0.9282 | 0.9153 |
| day1 | 10 | 19561 | 0.9612 | 0.9346 | 0.9273 |
| day2 | 8 | 15038 | 0.9512 | 0.9284 | 0.9313 |
| day2 | 9 | 14913 | 0.9767 | 0.9675 | 0.9688 |
| day2 | 10 | 14352 | 0.9741 | 0.9644 | 0.9633 |
| day2 | 11 | 6124 | 0.9930 | 0.9956 | 0.9890 |

## Notes and caveats

- **New sessions need calibration.** Per-session normalization uses each day's own mean/std. For a new day, record a short calibration block (rest plus a few gestures) to compute these stats. Unless that day is also added to training, the model uses only the shared weights for it. To add a day: put its files next to the others, add it to `SESSIONS` in `emg_data.py`, then run `train_svm.py` and `evaluate_svm.py`.
- **Temporal features are causal.** Moving averages only look at past samples, so the model can run in real time. It needs about 400 samples of history to fill the longest window.
- Normalization stats are computed from training repetitions only. CV folds share those stats (they use no labels), so CV scores can be very slightly optimistic. The test set is unaffected.

## Metric definitions

- **Accuracy**: fraction of samples classified correctly. Inflated by the large rest class.
- **Balanced accuracy**: mean recall across classes. Not affected by class imbalance.
- **Precision**: of the samples predicted as class *k*, the fraction that truly are *k*.
- **Recall / sensitivity**: of the samples that truly are class *k*, the fraction predicted as *k*.
- **Specificity**: of the samples that are not class *k*, the fraction correctly not predicted as *k*.
- **F1**: harmonic mean of precision and recall. Macro = unweighted mean over classes; weighted = mean weighted by class support.
- **Cohen's kappa**: agreement corrected for chance (0 = chance, 1 = perfect).
- **MCC**: correlation between true and predicted labels (−1 to 1). Robust to imbalance.
- **ROC AUC**: one-vs-rest area under the ROC curve from SVM decision scores. Measures how well the scores rank class *k* above the others, independent of the threshold.
