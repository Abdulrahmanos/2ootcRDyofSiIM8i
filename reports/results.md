# Reproduced baseline results

Binary F1 treats `flip` as the positive class. Grouped values use pooled out-of-fold predictions.

| Evaluation | Flip F1 | Precision | Recall |
| --- | ---: | ---: | ---: |
| Always flip, grouped validation | 0.6539 | 0.4858 | 1.0000 |
| Logistic regression, grouped validation | 0.8568 | 0.8276 | 0.8881 |
| Logistic regression, supplied test | 0.9772 | 0.9929 | 0.9621 |

Grouped F1 gain over the always-flip reference: **0.2029**, or **20.29 percentage points** on the 0-100 F1 scale.

Grouped fold mean: **0.8586**; sample standard deviation: **0.0517**. This standard deviation is not a confidence interval.

Grouped confusion counts: TN=1015, FP=215, FN=130, TP=1032.

The supplied-test score exceeds the pooled grouped score by 0.1204. All 65 filename video IDs overlap the supplied folders. These protocols evaluate different frames and fit on different amounts of data, so the gap is not an isolated causal estimate of leakage. Filename IDs are assumed to identify videos consistently; recording-session independence has not been established.

Source: [run metadata](metrics.json), [OOF predictions](validation_predictions.csv), [test predictions](test_predictions.csv), [fold scores](fold_metrics.csv), and [data manifest](dataset_manifest.csv). Regenerate by running the notebook from top to bottom. The exported predictions allow score recalculation without the source images.
