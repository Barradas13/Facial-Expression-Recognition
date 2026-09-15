# Methodology

The development of the facial expression recognition system will follow the OSEMN data science lifecycle, divided into five principal stages: **Obtain, Scrub, Explore, Model, and iNterpret**. Each stage contains a set of computational and experimental activities responsible for transforming raw video recordings into a deployable facial-expression recognition system.

## 1. Obtain

The Obtain stage will focus on acquiring the data necessary to train the facial expression recognition models. The collected data will consist of video recordings of multiple participants performing predefined facial expressions. Initially, the project will consider three primary classes:

* **Neutral**
* **Smiling**
* **Sad**

Additional expressions may be incorporated in later iterations of the project depending on the quantity and quality of the collected data.

### 1.1 Data Gathering

A standardized recording protocol will be defined so that participants record themselves under controlled but varied conditions. Each participant will record multiple videos for each expression, attempting to maintain the requested expression for a period of time rather than producing only a single facial pose.

The dataset should include variation in:

* participants;
* face shape and physiological characteristics;
* gender and age;
* illumination;
* camera distance;
* head orientation;
* background;
* expression intensity;
* recording duration.

This variation is important because the objective is not merely to memorize the participants used during training, but to develop a model capable of generalizing to previously unseen individuals.

### 1.2 Manual Annotation

During data collection, each video will receive an expression label. A dedicated annotation tool may be developed using OpenCV, following the same principle used in the previous blink project, where a participant could generate annotations while recording. The annotations will associate temporal portions of the video with a specific expression.

Possible labels include:

$$
y \in \{\text{neutral},\text{smiling},\text{sad}\}
$$

The annotation process should also permit later correction of incorrectly synchronized or ambiguous labels.

### 1.3 Image Processing and Facial Landmark Extraction

Each video will be converted into a sequence of frames. For every frame, a face detector and facial landmark predictor will be used to identify the coordinates of characteristic regions of the face.

The project may initially use **Dlib's 68-point facial landmark model**, which is particularly appropriate because it provides landmarks covering the eyebrows, eyes, nose, mouth, jaw, and other facial regions. The existing project already uses Dlib's facial landmark predictor and OpenCV for video processing.

For each frame \(t\), the system will obtain a set of facial landmarks:

$$
F_t =
\left[
(x_1,y_1),
(x_2,y_2),
\dots,
(x_{68},y_{68})
\right]
$$

These coordinates constitute the raw numerical representation of the face.

### 1.4 Feature Construction

Rather than relying exclusively on raw coordinates, the project may calculate geometric features describing facial configuration. Examples include:

* mouth width and height;
* mouth aspect ratio;
* eyebrow-to-eye distances;
* eye aspect ratio;
* distances between facial landmarks;
* angles between facial landmarks;
* relative movement of the mouth and eyebrows.

The project can therefore investigate two representations, similarly to the previous blink work:

**Approach A — Raw/normalized landmarks**

$$
X_t =
[x_1,y_1,x_2,y_2,\dots,x_{68},y_{68}]
$$

**Approach B — Derived geometric features**

$$
X_t =
[f_1,f_2,\dots,f_n]
$$

Comparing these approaches can reveal whether the classifier performs better using the complete facial geometry or a smaller set of interpretable measurements.

### 1.5 Data Storage

The extracted information will be stored in a structured dataset. A possible record format is:

| frame | participant | video    | timestamp | landmarks/features | label   |
| ----- | ----------- | -------- | --------- | ------------------ | ------- |
| 001   | P01         | smile_01 | 0.00 s    | ...                | smiling |
| 002   | P01         | smile_01 | 0.07 s    | ...                | smiling |
| 003   | P01         | smile_01 | 0.13 s    | ...                | smiling |

The dataset should preserve participant identity and video identity so that later train/test splitting can be performed without accidentally allowing frames from the same recording to appear in both sets.

---

# 2. Scrub

The Scrub stage will transform the extracted data into a consistent and reliable dataset suitable for statistical analysis and machine learning.

### 2.1 Missing-Value Treatment

Frames in which a face cannot be detected or landmarks cannot be reliably extracted will be identified.

Possible strategies include:

* removing isolated invalid frames;
* interpolating short gaps when appropriate;
* carrying information from neighboring frames;
* marking longer missing sequences as invalid.

The selected strategy should be documented because excessive interpolation could artificially modify facial movements.

### 2.2 Outlier Detection

Statistical techniques will be used to identify abnormal landmark coordinates or feature values. Outliers may originate from incorrect face detection, partial occlusion, extreme head movement, or landmark prediction failures.

The previous blink project used explicit treatment of outliers in both landmark coordinates and derived EAR values. A similar principle can be adopted here, but the limits should be determined from the facial-expression dataset rather than copied from the blink project.

### 2.3 Face and Head-Pose Normalization

Raw facial coordinates depend strongly on the position and scale of the face in the image. A participant moving closer to the camera should not automatically appear to be producing a different expression.

Therefore, the landmarks will be normalized using facial reference points. One possible method is to translate the coordinates relative to a stable facial reference point and normalize their scale according to a characteristic facial distance.

For example:

$$
x'_i = \frac{x_i-x_c}{d},
\qquad
y'_i = \frac{y_i-y_c}{d}
$$

where \((x_c,y_c)\) is a facial reference point and \(d\) is a normalization distance.

This allows the model to focus more strongly on facial deformation than on absolute position in the image.

### 2.4 Label Validation

The annotations will be examined to detect:

* incorrectly labeled frames;
* transitions between expressions;
* ambiguous frames;
* periods in which the participant is not performing the requested expression;
* annotation delays.

This is particularly important for temporal models because a small labeling delay can shift the expression relative to the facial landmarks.

The previous blink project encountered exactly this type of annotation synchronization problem and observed that manual annotations could contain a small temporal delay.

### 2.5 Dataset Balancing

The number of samples belonging to each expression will be analyzed. If one class contains substantially more observations than another, balancing techniques may be considered to reduce class imbalance.

Possible strategies include controlled sampling, class weighting, or collection of additional recordings for underrepresented expressions.

---

# 3. Explore

The Explore stage will be used to understand the structure of the dataset before selecting the final machine-learning model.

### 3.1 Descriptive Statistics

Statistical measures will be calculated for the principal facial features, including:

* mean;
* median;
* standard deviation;
* minimum;
* maximum;
* quartiles.

These statistics can reveal how strongly different facial measurements change between expressions.

### 3.2 Visualization of Facial Features

Graphs will be created to investigate differences between classes.

Examples include:

* boxplots of mouth width for each expression;
* distributions of eyebrow distances;
* scatter plots between facial measurements;
* feature correlation matrices;
* time-series plots of facial measurements;
* average normalized facial landmarks by class.

For example, a temporal feature such as mouth opening can be represented as:

$$
f(t)
$$

and compared between neutral, smiling, and sad recordings.

### 3.3 Temporal Analysis

Because facial expressions develop over time, the temporal behavior of the landmarks will be investigated.

For example:

$$
X =
[X_{t-k+1},X_{t-k+2},\dots,X_t]
$$

may represent a temporal window containing \(k\) consecutive frames.

The project can investigate whether five frames, ten frames, fifteen frames, or another window length provides the best representation of facial-expression dynamics.

This directly extends the temporal-window approach used in the blink project, where five consecutive frames constituted one classification input.

### 3.4 Feature Selection

Correlation analysis, feature importance, statistical analysis, or dimensionality-reduction techniques may be used to identify which facial measurements contribute most strongly to expression classification.

The goal is to determine whether the entire facial landmark representation is necessary or whether a smaller subset—particularly involving the mouth, eyebrows, and eyes—is sufficient.

### 3.5 Inter-Class Analysis

The distributions of the three classes will be compared to determine which expressions are most easily distinguishable.

Particular attention should be given to:

$$
\text{neutral} \leftrightarrow \text{smiling}
$$

$$
\text{neutral} \leftrightarrow \text{sad}
$$

$$
\text{smiling} \leftrightarrow \text{sad}
$$

This analysis can guide feature engineering and model selection.

---

# 4. Model

The Model stage will transform the processed temporal facial data into predictive models capable of classifying facial expressions.

## 4.1 Temporal Window Construction

The fundamental input to the model will be a sequence of consecutive frames rather than an isolated frame.

For example, using a window of five frames:

$$
W_t =
[X_{t-4},X_{t-3},X_{t-2},X_{t-1},X_t]
$$

Each window will receive one target label:

$$
y_t \in
\{\text{neutral},\text{smiling},\text{sad}\}
$$

The window size will be treated as an experimental parameter.

Different configurations may be evaluated, such as:

$$
k \in \{5,10,15,30\}
$$

depending on the recording frame rate.

## 4.2 Input Representations

At least two representations will be investigated.

### Representation A — Landmark Coordinates

The model receives normalized coordinates of the facial landmarks over the temporal window.

For example:

$$
X =
[
x_1,y_1,\dots,x_{68},y_{68}
]
$$

for each frame.

### Representation B — Engineered Facial Features

The model receives a smaller set of geometric features derived from the landmarks.

For example:

$$
X =
[
\text{mouth width},
\text{mouth height},
\text{eyebrow distance},
\text{eye ratio},
\dots
]
$$

The performances of the two representations will be compared.

## 4.3 Candidate Machine-Learning Models

The initial experiments can use classical supervised-learning algorithms similar to those used in the blink project:

* Decision Tree;
* Random Forest;
* K-Nearest Neighbors;
* Logistic Regression.

The previous study used exactly these four classifiers and implemented them with Scikit-Learn.

Random Forest should be an especially useful baseline because the previous facial-landmark experiment achieved strong performance with that model.

However, the project should not assume beforehand that Random Forest will be the best solution. Performance must be experimentally determined.

## 4.4 Sequential Models

After establishing a classical-machine-learning baseline, the project may investigate models specifically designed for temporal information, such as:

* recurrent neural networks;
* LSTM;
* GRU;
* temporal convolutional models.

These should be considered an extension rather than a mandatory first implementation. The classical models provide an important baseline and may have substantially lower computational and deployment complexity.

## 4.5 Training and Validation

The dataset will be divided into training and testing subsets while preserving participant independence.

The most important principle will be:

> **A participant appearing in the test set should not appear in the training set.**

Otherwise, the model could learn characteristics of a person's face rather than characteristics of facial expressions.

A participant-level cross-validation strategy can therefore be adopted, following the principle used in the blink project. The previous work used Leave-One-Out validation, training on six participants and testing on the seventh, and repeated this process for all participants.

For a larger dataset, alternatives such as GroupKFold can also be investigated, where the participant identifier is used as the grouping variable.

## 4.6 Hyperparameter Optimization

The principal hyperparameters of each model will be optimized using the training data.

Possible techniques include:

* Grid Search;
* Random Search;
* cross-validation.

The test set will remain isolated until the final evaluation.

## 4.7 Evaluation Metrics

The models will be evaluated using:

$$
\text{Accuracy}
$$

$$
\text{Precision}
$$

$$
\text{Recall}
$$

$$
F1\text{-Score}
$$

as well as a multiclass confusion matrix.

The previous project already used accuracy, precision, recall, and F1-score as its principal evaluation metrics.

Because this project has multiple classes, macro-averaged metrics should also be considered so that performance on minority classes is not hidden by the most frequent class.

## 4.8 Real-Time Inference

After selecting the best model, the inference pipeline will perform:

$$
\text{Video/Webcam}
\rightarrow
\text{Face Detection}
\rightarrow
\text{Landmarks}
\rightarrow
\text{Normalization}
\rightarrow
\text{Temporal Window}
\rightarrow
\text{Model}
\rightarrow
\text{Expression}
$$

The output may be represented as:

```python
{
    "label": "smiling",
    "confidence": 0.94,
    "timestamp": 12.47
}
```

or, for a complete stream:

```python
[
    {"label": "neutral", "confidence": 0.91},
    {"label": "smiling", "confidence": 0.95},
    {"label": "smiling", "confidence": 0.97},
    ...
]
```

---

# 5. iNterpret

The iNterpret stage will evaluate the practical meaning of the trained model and transform the research implementation into a reusable software system.

## 5.1 Model Interpretation

The classification results will be analyzed to identify:

* which expressions are most frequently confused;
* which facial features are most informative;
* how performance changes between participants;
* how illumination affects predictions;
* how head movement affects predictions;
* how expression intensity affects predictions;
* how processing speed changes under real-time conditions.

Feature importance or permutation importance may be used for interpretable models such as Random Forest.

## 5.2 Error Analysis

Incorrect predictions will be examined individually.

Typical error categories may include:

* neutral faces predicted as sad;
* weak smiles predicted as neutral;
* exaggerated expressions correctly classified while subtle expressions fail;
* head rotation causing incorrect predictions;
* occlusion caused by glasses, hands, hair, or other objects;
* incorrect landmark detection;
* transitions between expressions.

This qualitative analysis is important because high overall accuracy does not necessarily imply reliable behavior in every practical situation.

## 5.3 Real-Time Performance

The inference pipeline will be evaluated not only by predictive quality but also by computational performance.

Relevant measurements include:

$$
\text{Inference Time / Window}
$$

$$
\text{Frames Per Second}
$$

$$
\text{End-to-End Latency}
$$

The previous blink implementation explicitly evaluated processing time and reported approximately 17 ms per model evaluation on its test computer, demonstrating that computational performance can be measured alongside classification accuracy.

## 5.4 Model Export

The final trained model will be serialized into a portable model artifact.

For example:

```text
models/
    facial_expression_model.joblib
```

The landmark-processing configuration and preprocessing parameters should also be stored so that the exact same transformations used during training are applied during inference.

## 5.5 Python Library

A principal objective of the project will be to make the trained system accessible through a simple Python interface.

A possible API is:

```python
import facialrecognition

result = facialrecognition.recognize(0)
```

where:

```text
0
```

represents the default webcam.

Alternatively:

```python
result = facialrecognition.recognize("video.mp4")
```

could process a video file.

The API could return either the complete prediction sequence or a real-time iterator:

```python
for result in facialrecognition.recognize(0):
    print(result.label, result.confidence)
```

This architecture separates the machine-learning implementation from the application using it.

## 5.6 Docker Deployment

A Docker image will provide an alternative deployment mechanism for users who do not want to manually configure the Python environment.

A conceptual interface could be:

```bash
docker run --rm facial-expression-recognition video.mp4
```

or, for webcam processing, a platform-specific camera configuration could be provided.

The container should include:

* the Python runtime;
* model weights;
* facial landmark model;
* Python dependencies;
* inference code;
* command-line interface.

This makes the system reproducible and substantially reduces installation differences between users.

## 5.7 Final Software Architecture

The final project can therefore be organized into three layers:

```text
                 ┌───────────────────────────┐
                 │      Input Interface       │
                 │                           │
                 │   Webcam / Video File     │
                 └────────────┬──────────────┘
                              │
                              ▼
                 ┌───────────────────────────┐
                 │   Computer Vision Layer   │
                 │                           │
                 │ Face Detection            │
                 │ Facial Landmarks          │
                 │ Normalization             │
                 │ Temporal Windows          │
                 └────────────┬──────────────┘
                              │
                              ▼
                 ┌───────────────────────────┐
                 │     Machine Learning      │
                 │                           │
                 │ Trained Expression Model │
                 │                           │
                 │ neutral / smiling / sad  │
                 └────────────┬──────────────┘
                              │
                              ▼
                 ┌───────────────────────────┐
                 │      Output Interface     │
                 │                           │
                 │ label + confidence        │
                 │ timestamp / sequence      │
                 └───────────────────────────┘
```

## 5.8 Reproducibility

All experiments should be stored in a reproducible repository containing:

```text
facial-expression-recognition/
│
├── data/
├── notebooks/
│   ├── 01_obtain.ipynb
│   ├── 02_scrub.ipynb
│   ├── 03_explore.ipynb
│   ├── 04_model.ipynb
│   └── 05_interpret.ipynb
│
├── src/
│   ├── extraction/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   └── inference/
│
├── annotation/
├── models/
├── tests/
├── Dockerfile
├── pyproject.toml
├── README.md
└── LICENSE
```

The separation of notebooks, extraction software, model artifacts, and inference code will make it possible to distinguish experimentation from the final reusable product.


Facial landmarks→Facial features + temporal window→Expression