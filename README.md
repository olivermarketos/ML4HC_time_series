# ML4HC_time_series

Below we provide an outline of the order to run the different jupyter notebooks, which correspond to the different section laid out in the assignmenet brief.

## Initial Steps

1. Clone the Repository
```
git clone https://github.com/olivermarketos/ML4HC_time_series.git
cd ML4HC_time_series
```

2. Set Up a Virtual Environment

Option A: Using Python venv:
```
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```
Option B: Using Conda (recommended):
```
conda create --name yourenv python=3.11
conda activate yourenv
```

3. Install Requirements
```
pip install -r requirements.txt
```
---

## Question 1

#### 1.1. Data pre-processing
To preprocess the raw data, run process_data.py. By default it will process into a time grid format (used for majority of questions). To change format to that used in Question 3.2b, change the format tag at the bottom of the file to `"time_tuple"`

#### 1.2 Data visualisation and exploration
For some data vizualisation, run `data_viz.ipynb`

#### 1.3 Impute missing values
For forward filling followed by median filling in the preprocessed data files from 1.1 run `fill_nan.ipynb`

---

## Question 2

### 2.1 Classic ML Models
#### 2.1.1. 
To compute the 41 features vectors, run `Q2_1_1_features.ipynb`
To train and evaluate the models on the 41 features vectors, run `Q2_1_1_models.ipynb`

#### 2.1.2.
To compute the 348 features vectors,` Q2_1_2_features_ts.ipynb`
To train and evaluate the models on the 348 features vectors, run `Q2_1_2_models_ts.ipynb`

(Without tsfresh features, only statistical features (min, max, skew, std, mean)
    
To compute the 189 features vectors, `Q2_1_2_features.ipynb`
To train and evaluate the models on the 189 features vectors, run `Q2_1_2_models.ipynb`

The most important features of the random forest model trained on the 189 features was used to create the textual summary of patients for LLM.
The tsfresh features were not used because too complex for the LLM
)

### 2.2 Recurrent Neural Networks
To create the trained model and the according metrics, run the `Q2_2.py file`.
The `Q2_2.ipynb` file has the same content.

### 2.3 Transformers
`Q2_3_data_processing.ipynb` to process the data into format Time-Grid or (Time, Measurement, Value) triplets, to be used for each of the transformer architectures in this section.
`Q2_3_transformers.ipynb` to train and evaluate either of the transformer architectures. It makes use of addtional scripts `Transformers.py` (transformer architectures, dataset classes etc) and `helper_funcs.py` (data loading, model loading, training, evaluation etc).

 > **note**: this section and section Q3 make use of a config file, loaded at the beginning of the notebook. `config.yaml` provides an example file. It contains the hyperparameters, output directories and which m

---

## Question 3
`Q3_Representation_learning.ipynb` contains all the code needed to answer this section. It makes use of addtional scripts `Transformers.py` and `helper_funcs.py`

---

## Question 4

#### 4.1
To generate the predictions of the test set by prompting the LLM, run `Q4_pred_c.ipynb`

To evaluate these predictions, run `Q4_1.ipynb`

#### 4.2
To generate the LLM embeddings of the training and test set, run `Q4_emb_a_c.ipynb`
To train on these LLM embeddings and evaluate logistic regression model, and also to generate t-sne on these LLM embeddings , run `Q4_2_2.ipynb`

#### 4.3
To create the trained model and the according metrics, run the `Q4_3.py file`.
The `Q4_3.ipynb` file has the same content.

