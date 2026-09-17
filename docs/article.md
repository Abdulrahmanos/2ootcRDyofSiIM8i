# What a page-flip classifier taught me about evaluating video frames

*A reproducible baseline from an Apziva AI Residency project.*

The first result that needs explaining in this project is the difference between two F1 scores: **0.9772 on the supplied test split** and **0.8568 when holding out video IDs**.

Both came from the same model design. They describe different evaluation conditions. Understanding those conditions is a useful part of building a classifier that someone else can assess.

This project was developed in the context of the [Apziva AI Residency](https://www.apziva.com/). The task is to classify a single image as showing a page flip or no page flip. It supports one part of a document-scanning workflow; the implementation here is a baseline study, with sequence detection and deployment left for future experiments.

The walkthrough below follows the [executable notebook](https://github.com/Abdulrahmanos/2ootcRDyofSiIM8i/blob/main/01_monreader_baseline.ipynb). Code excerpts use variables defined there; run the full notebook from the project root to reproduce the results.

## 1. Start by defining the prediction

The input is one document image. The output is one of two labels: `notflip = 0` or `flip = 1`. The project uses binary F1 with `flip` as the positive class.

Precision measures how many predicted flips are real flips. Recall measures how many real flips are detected. F1 combines them as `2 TP / (2 TP + FP + FN)`. A model can catch every flip and still perform poorly if it also calls many stationary pages flips. [F1 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).

Those errors could matter to a capture workflow, but this experiment does not measure their effect on scanning quality or user experience.

## 2. Audit the data before fitting

The supplied data contains 2,989 JPEG images: 2,392 in training and 597 in testing. Training contains 1,162 flip and 1,230 notflip images; testing contains 290 flip and 307 notflip images.

The inventory decodes every file, reads its dimensions, extracts video ID and frame number from its filename, and calculates a SHA-256 file hash. All images decode successfully and are 1080 by 1920 RGB images. There are no repeated file hashes.

That last finding has a narrow meaning: the files are not byte-for-byte duplicates. Nearby video frames can still look almost identical.

## 3. Check what the split separates

The filenames follow `VideoID_FrameNumber.jpg`. All 65 video IDs occur in both the supplied training and testing folders.

Assuming those IDs consistently identify source videos, the supplied test set contains held-out frames from videos represented during training. A classifier may encounter familiar books, camera angles, or backgrounds. This is a different question from classifying frames from an excluded video.

To examine that second question, the notebook applies five-fold grouped validation inside the training folder:

```python
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(cv.split(train, train["target"], groups=train["video_id"]))

for fit_idx, val_idx in folds:
    fit_videos = set(train.iloc[fit_idx]["video_id"])
    val_videos = set(train.iloc[val_idx]["video_id"])
    assert fit_videos.isdisjoint(val_videos)
```

The splitter keeps groups separate while attempting to preserve class proportions. [StratifiedGroupKFold documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

This still has limits. Different video IDs could share a book or recording session. The filenames alone cannot establish that those conditions are independent.

## 4. Use a small, explicit baseline

Each image is converted to grayscale, resized to 36 by 64 pixels, divided by 255, and flattened into 2,304 values. This keeps the supplied portrait aspect ratio, but discards detail and makes the representation sensitive to alignment.

The model is a pipeline of standardization and regularized logistic regression:

```python
baseline = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        C=0.01, solver="liblinear", max_iter=2000, random_state=42
    ),
)
```

A fresh copy is fitted inside each fold. That also fits the scaler using only the fitting subset, so validation images do not influence its statistics. Pipelines help keep fitted preprocessing tied to the correct training data. [scikit-learn guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

The model settings and 0.5 threshold were fixed before the first evaluation. There was no hyperparameter search. An always-flip classifier provides a reference on the same folds; its perfect recall makes clear why recall alone is insufficient.

## 5. Read the results with their protocol

| Model and evaluation | Flip F1 | Precision | Recall |
| --- | ---: | ---: | ---: |
| Always flip, grouped validation | 0.6539 | 0.4858 | 1.0000 |
| Logistic regression, grouped validation | 0.8568 | 0.8276 | 0.8881 |
| Logistic regression, supplied test | 0.9772 | 0.9929 | 0.9621 |

The grouped score pools out-of-fold predictions for all 2,392 training frames. Each prediction comes from a model that excluded that frame's video ID. The learned model improves F1 by 0.2029 over the reference under this protocol.

Individual fold F1 scores range from 0.8040 to 0.9412. Their mean is 0.8586, with sample standard deviation 0.0517. This variation is useful context; the standard deviation is not a confidence interval, and the pooled score is not an equal-weight average across videos.

For the supplied benchmark, the same model design is fitted on all training images and evaluated on the 597 test images. Its higher F1 should be reported alongside the known video overlap. The 0.1204 difference cannot be attributed entirely to that overlap: the evaluated frames and training-set sizes also change. This comparison motivates careful interpretation rather than providing a controlled estimate of leakage.

![Scores labeled by evaluation protocol](https://github.com/Abdulrahmanos/2ootcRDyofSiIM8i/blob/main/reports/f1_comparison.png)

## 6. Examine mistakes before proposing a bigger model

The grouped confusion matrix contains 1,032 true positives, 1,015 true negatives, 215 false positives, and 130 false negatives. The notebook displays a deterministic sample of the 345 errors.

Some sampled frames contain visually subtle page movement or hand motion. These are candidates for further investigation, not established causes of the model's errors. Reviewing a handful of examples cannot determine how common a failure pattern is or resolve possible label ambiguity.

A useful next experiment would compare richer image features using the same video groups. A separate temporal experiment could define labeled sequence windows and test whether motion context improves sequence-level detection. Neither result can be claimed from this single-image baseline.

## 7. Make the result inspectable

The notebook exports the image manifest, fold membership, row-level predictions, fold scores, model settings, package versions, and hashes of the manifest and notebook code. The saved predictions allow readers to recalculate F1 without access to the images:

```python
oof = pd.read_csv("reports/validation_predictions.csv")
f1_score(oof["target"], oof["logistic_pixels_prediction"])
```

The fitted model is saved separately with its label mapping, threshold, and resize settings. A prediction helper demonstrates reuse of the same preprocessing. Calibration and target-device latency remain unmeasured.

## What this experiment establishes

This baseline performs better than an always-flip reference on held-out video IDs within the available dataset. Its results also show why a supplied test score needs a clear account of how the data was split.

The next stage is to improve the model under a stable evaluation protocol and collect an independent evaluation set with known capture conditions. Until then, the contribution is a reproducible experiment and a documented basis for further work.

Project context: [Apziva](https://www.apziva.com/). Code and evidence: [notebook](https://github.com/Abdulrahmanos/2ootcRDyofSiIM8i/blob/main/01_monreader_baseline.ipynb), [results](https://github.com/Abdulrahmanos/2ootcRDyofSiIM8i/blob/main/reports/results.md), and [evidence index](https://github.com/Abdulrahmanos/2ootcRDyofSiIM8i/blob/main/reports/README.md).

GitHub repository: [Abdulrahmanos/2ootcRDyofSiIM8i](https://github.com/Abdulrahmanos/2ootcRDyofSiIM8i).

*Article draft. The repository is currently private; its links require access.*
