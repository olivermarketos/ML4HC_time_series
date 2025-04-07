# ML4HC_time_series


## Question 1

1.1. 
To preprocess the raw data, run process_data.py

1.2 For some data vizualisation, run data_viz.ipynb

1.3 For forward filling followed by median filling in the preprocessed data files from 1.1 run fill_nan.ipynb

## Question 2

2.1.1. 
To compute the 41 features vectors, run Q2_1_1_features.ipynb
To train and evaluate the models on the 41 features vectors, run Q2_1_1_models.ipynb

2.1.2.
To compute the 348 features vectors, Q2_1_2_features_ts.ipynb
To train and evaluate the models on the 348 features vectors, run Q2_1_2_models_ts.ipynb

(Without tsfresh features, only statistical features (min, max, skew, std, mean)
    
To compute the 189 features vectors, Q2_1_2_features.ipynb
To train and evaluate the models on the 189 features vectors, run Q2_1_2_models.ipynb

The most important features of the random forest model trained on the 189 features was used to create the textual summary of patients for LLM.
The tsfresh features were not used because too complex for the LLM
)

## Question 3

## Question 4

4.1
To generate the predictions of the test set by prompting the LLM, run Q4_pred_c.ipynb

To evaluate these predictions, run Q4_1.ipynb

4.2
To generate the LLM embeddings of the training and test set, run Q4_emb_a_c.ipynb
To train on these LLM embeddings and evaluate logistic regression model, and also to generate t-sne on these LLM embeddings , run Q4_2_2.ipynb

