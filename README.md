# HEART-DISEASE-PREDICTION-PROJECT

### Project Overview.

This project focuses on predicting the likelihood of heart disease in individuals based on a set of medical features. The objective is to develop a machine learning model that can predict heart disease using different classification algorithms, including Logistic Regression, Random Forest,SVM and XGBoost.

The dataset used contains various health-related metrics such as age, cholesterol levels, blood pressure, and more.Models were trained and compared based on their predictive performance, and hyperparameter tuning was performed to optimize their performance.

### Project Description.

The heart disease prediction model aims to assist healthcare professionals in identifying patients who may be at risk of developing heart disease. Early detection is crucial in preventing severe health complications, and machine learning models can provide timely and accurate predictions.

### Installation.

To run this project, you'll need the following Python libraries:

- Python 3.x
  
- Numpy
  
- Pandas
  
- Scikit-learn
  
- XGBoost
  
- Matplotlib (for visualizations)


### Dataset Description.

The dataset used for this project contains multiple features representing medical metrics that are important in determining heart disease risk. These include:

- Age
  
- Sex
  
- Chest pain type (4 values)
  
- Resting blood pressure

- Cholesterol levels
  
- Fasting blood sugar > 120 mg/dl (binary)
  
- Resting electrocardiographic results (0, 1, 2)
  
- Maximum heart rate achieved
  
- Exercise-induced angina (binary)
  
- Oldpeak (ST depression induced by exercise)
  
- Slope of the peak exercise ST segment
  
- Number of major vessels (0-3)
  
- Thalassemia
  
- The target variable is whether the individual has heart disease (1) or not (0).

### Preprocessing.

Before training the models:

1. Handling Missing Values : There were no missing values.

2. Feature Engineering : Grouping age into various age groups and grouping cholestral level depending on the groups available.

3. Droping unecessary columns : ie age and chol.
   
4. Feature Encoding: Categorical variables like chest pain type, thalassemia, and electrocardiographic results were encoded using one-hot encoding.

5. Feature Scaling: Features like  trestbps ,thalach and oldpeak were scaled to improve the performance of gradient-based algorithms.
   
6. Splitting Data: The dataset was split into training (80%) , validation(20%) and testing (20%) sets to evaluate model generalization.

### Model training and Evaluation.

  #### XGBoost:
  
  - A gradient boosting algorithm optimized fror performance.
    
  - initial hyperparameters:                  
            
              
        - colsample_bytree=1.0,
        - learning_rate=0.1,
        - max_depth=7,
        - n_estimators=150,
        - random_state=42

  - Train Accuracy : 0.99
    
  - Recall
    
    - class 0 : 0.99
    - class 1 : 1.00
    
  - precision
    
    - class 0 : 1.00
    - class 1 : 0.99
   
  - Test Accuracy : 0.99

  - Recall
    
    - class 0 : 0.98
    - class 1 : 1.00

  - Precision
    
    - class 0 : 1.00
    - class 1 : 0.98 

  #### Random Forest :
  - An ensemble method that builds multiple decision tress and average results.

  - Initial parameters :
                        
        - n_estimators=300,
        - min_samples_split=2,
        - min_samples_leaf=1,
        - max_features='log2',
        - max_depth=None,
        - random_state=42

  - Train accuracy : 0.97

  - Recall
    - class 0 : 0.99
    - class 1 : 0.96

  - Precision
    - class 0 : 0.96
    - class 1 : 0.99
   
  -Test accuracy : 0.99

  - Recall
    - class 0 : 0.98
    - class 1 : 1.00

  - Precision
    - class 0 : 1.00
    - class 1 : 0.98
   
   #### Logistic Regression:

   - A basic classification model used as a baseline.
     
   - Initial parameters :
     
          - C=1,
          - solver='lbfgs',
          - random_state=42
            
   -Trian accuracy : 0.87

   - Recall
     - class 0 : 0.81
     - class 1 : 0.89

  - Precision
    - class 0 : 0.93
    - class 1 : 0.84

  #### SVM

  - kernel
    
        - linear.
    
  - Train accuracy : 0.80

  - Recall
    - class 0 : 0.74
    - class 1 : 0.86
   
  - Precission
    - class 0 : 0.84
    - class 1 : 0.77


### Hyperparameter Tuning

1. XGBoost was fine-tuned using GridSearchCV and cross validation to optimize the hyperparameters.
The final tuned hyperparameters were :

        {
            'colsample_bytree': 1.0,
            'gamma': 0,
            'learning_rate': 0.2,
            'max_depth': 5,
            'min_child_weight': 1,
            'n_estimators': 200,
            'subsample': 0.8
        }

After tuning the test accuracy was 0.97 recall for class 0 and 1 were 0.98 and 0.97 respetively and Precision for class 0 and 1 were 0.97 and 0.98 respectively.

2. **Random Forest** was fine tuned using GridSearchCV and cross validation to optimize the hyperparameters.

 The final tuned hyperparamets were :
 
     {
       'max_depth': None, 
       'max_features': 'log2', 
       'min_samples_leaf': 1, 
       'min_samples_split': 2, 
       'n_estimators': 100
       
      }

After tuning the test accuracy was 0.99 , recall for class 0 and 1 were 0.98 and 1.0 respectively and Precision for class 0 and 1 were 1.0 and 0.98 respectively.


#### Model Performance:📜

-After tuning, Random Forest achieved slightly better results in terms of test accuracy (0.99) compared to XGBoost (0.9756). Random Forest was able to generalize better on unseen data.

**Recall & Precision:**

-Random Forest outperformed XGBoost in terms of recall for class 1 (heart disease detection).score of 1.00 indicates that Random Forest identified all true positives (people with heart disease) without missing any cases. This is a critical factor in medical diagnosis, where false negatives can have serious consequences. On the other hand, XGBoost has a slightly lower recall for class 1 (0.97), meaning it might miss some positive cases.

### Future Work 🛠

Possible areas for improvement:

1. Model Deployment: Deploy the trained RandomForest model as a web API for real-time heart disease prediction or directly into* hospital systems.

2. Additional Features: Incorporate more patient history and lifestyle data.
   
3. Advanced Models: Experiment with deep learning models such as neural networks.



Author

Shisia Joy

For questions or support please contact me at [shisiajoy4@gmail.com] 😊



























  
