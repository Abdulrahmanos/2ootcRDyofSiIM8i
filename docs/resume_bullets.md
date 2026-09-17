# Resume wording grounded in the current experiment

Project title: **MonReader - Page-Flip Classification**

Tools: **Python, scikit-learn, NumPy, pandas, Pillow, Matplotlib**

These are drafts for the projects section and the Apziva experience entry. They follow an outcome, measurement, and method structure. No existing resume file was provided, so the text is kept here for insertion.

## Project section

- Built a page-flip classification baseline with **0.8568 F1 across 2,392 out-of-fold frame predictions** by implementing a Python/scikit-learn pipeline and validating across five disjoint video-ID folds.
- Improved grouped-validation F1 by **0.2029 over an always-flip reference (0.6539 to 0.8568)** using standardized grayscale pixel features and regularized logistic regression on the same folds.
- Made evaluation results checkable for a **2,989-image dataset** by auditing file hashes and video overlap, then exporting per-frame predictions, fold assignments, model settings, and environment metadata.

For a compact resume, use the first and third bullets; the second is an alternative when the reference comparison is more relevant than the artifact work.

## One bullet for the Apziva experience section

- Developed and evaluated a page-flip classification baseline with **0.8568 grouped-validation F1**, versus **0.6539** for an always-flip reference, using Python, scikit-learn, and five-fold validation that held out video IDs.

## Evidence and wording notes

| Claim | Evidence | Meaning |
| --- | --- | --- |
| 0.8568 F1 | [Validation metrics](../reports/validation_metrics.csv) and [OOF predictions](../reports/validation_predictions.csv) | Pooled binary F1 for `flip`, not accuracy or an equal-weight video average |
| Improvement of 0.2029 | [Generated comparison](../reports/results.md) | Absolute F1 difference against a fixed reference under the same grouped protocol |
| 2,989 audited images | [Manifest](../reports/dataset_manifest.csv) and [run metadata](../reports/metrics.json) | The total supplied dataset; 2,392 frames are used for grouped validation |
| Five disjoint video-ID folds | [Fold summary](../reports/fold_summary.csv) and [notebook](../01_monreader_baseline.ipynb) | Separation by filename video ID, not independently verified book/session separation |

The 0.9772 supplied-test F1 is documented in the project but omitted from these compact bullets because its video overlap needs additional context. The current evidence supports a baseline experiment, not claims of production deployment, measured business impact, or improved outcomes for users.
