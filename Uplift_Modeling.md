# Introduction
This homework is about "uplift modeling" (see the [wiki page](https://en.wikipedia.org/wiki/Uplift_modelling) for an introduction) which is basically a combination of machine learning + causual inference.

## Data
I am using sample of data from the [Criteo Uplift Prediction Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/).  You can read a more detailed description of the data there, and there is also a [research paper](https://drive.google.com/file/d/1JTKuzdl7xxQuLuwqfmZBsvhMarxCzHh7/view?usp=sharing) about uplift modeling using this dataset.  Note that the description says there are 11 features, but our dataset actually has 12 (I think the documentation is out of date)

There are two datasets: one for training/validation, and a separate one for testing. Each row in the data represents a different user. The training data has the following fields:
* 12 features - all continuous, named f0, f1, f2, ..., f11
* an indicator as to whether the user received treatment (treatment = 1, control = 0).  "Treatment" means the user was exposed to a specific advertising campaign.
* an indicator as to whether the user performed a website visit (vist = 1, no visit = 0)

The test data only has the 12 continuous features, it does not have the indicators for treatment or website visit.




```python
training_data_file = 'https://github.com/hangingbelay/spring2024/raw/main/homework_4_training_data.csv.gz'

test_data_file = 'https://github.com/hangingbelay/spring2024/raw/main/homework_4_test_data.csv.gz'
```


```python
# Loading the packages

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
```

## Hints and tips

* There is NO data cleaning or outlier analysis necessary for this dataset.

* I used the [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) module from scikit learn.

* The code needed to to train and evaluate a logistic regression model is shown in the class activity on April 24th.  This includes feature scaling as well.

*Note:* It is best practice to use the same scaling for all of your models / predictions
  
In any randomized experiment, it is good practice to ensure that the features are independent of the assignment to treatment or control.  One way to do this is to build a model using the treatment indicator as the response variable, and then measure the predictive power of this model.  If there is no predictive power (i.e. the model predicts no better/worse than a random prediction) then this is strong evidence that features are independent of assignment.


---
Using only the training data, I built a model with the treatment indicator as the response, and the 12 continuous features as predictors.  

I will checking the performance of this model as measured by AUC (area under roc curve)?



```python
train = pd.read_csv(training_data_file)

print(train.describe())

print('\n')

print(train.head())
```

                       f0              f1              f2              f3  \
    count  1000000.000000  1000000.000000  1000000.000000  1000000.000000   
    mean        19.611435       10.069976        8.446663        4.178904   
    std          5.377064        0.104578        0.299430        1.335977   
    min         12.616365       10.059654        8.214383       -8.398387   
    25%         12.616365       10.059654        8.214383        4.679882   
    50%         21.918222       10.059654        8.214383        4.679882   
    75%         24.420313       10.059654        8.725136        4.679882   
    max         26.745251       16.344187        9.051962        4.679882   
    
                       f4              f5              f6              f7  \
    count  1000000.000000  1000000.000000  1000000.000000  1000000.000000   
    mean        10.338641        4.028608       -4.148620        5.101948   
    std          0.342917        0.429742        4.575024        1.205538   
    min         10.280525       -8.912209      -29.306196        4.833815   
    25%         10.280525        4.115453       -6.699321        4.833815   
    50%         10.280525        4.115453       -2.411115        4.833815   
    75%         10.280525        4.115453        0.294443        4.833815   
    max         19.124973        4.115453        0.294443       11.998340   
    
                       f8              f9             f10             f11  \
    count  1000000.000000  1000000.000000  1000000.000000  1000000.000000   
    mean         3.933539       16.030668        5.333128       -0.170977   
    std          0.056715        7.029203        0.167315        0.022868   
    min          3.643174       13.190056        5.300375       -1.188243   
    25%          3.910792       13.190056        5.300375       -0.168679   
    50%          3.971858       13.190056        5.300375       -0.168679   
    75%          3.971858       13.190056        5.300375       -0.168679   
    max          3.971858       67.231876        6.473913       -0.168679   
    
                treatment           visit  
    count  1000000.000000  1000000.000000  
    mean         0.849532        0.047163  
    std          0.357530        0.211987  
    min          0.000000        0.000000  
    25%          1.000000        0.000000  
    50%          1.000000        0.000000  
    75%          1.000000        0.000000  
    max          1.000000        1.000000  
    
    
              f0         f1        f2        f3         f4        f5        f6  \
    0  25.940307  10.059654  8.214383  4.679882  10.280525  4.115453 -2.411115   
    1  20.386755  10.059654  9.036493  3.907662  10.280525  4.115453 -5.576414   
    2  22.833444  10.059654  8.214383  4.679882  10.280525  4.115453 -7.011752   
    3  13.514168  10.059654  8.281165  1.433141  10.280525  4.115453 -6.359690   
    4  15.675180  10.059654  8.214383  1.614662  10.280525  4.115453 -5.987667   
    
             f7        f8         f9       f10       f11  treatment  visit  
    0  4.833815  3.971858  13.190056  5.300375 -0.168679          1      0  
    1  4.833815  3.856829  13.190056  5.300375 -0.168679          1      0  
    2  4.833815  3.971858  13.190056  5.300375 -0.168679          1      0  
    3  4.833815  3.732154  45.715029  5.300375 -0.168679          1      1  
    4  4.833815  3.971858  13.190056  5.300375 -0.168679          1      0  
    


```python
test = pd.read_csv(test_data_file)

print(test.describe())

print('\n')

print(test.head())
```

                      f0             f1             f2             f3  \
    count  500000.000000  500000.000000  500000.000000  500000.000000   
    mean       19.624292      10.069864       8.446247       4.178380   
    std         5.378992       0.104810       0.299191       1.337044   
    min        12.616365      10.059654       8.214383      -6.361766   
    25%        12.616365      10.059654       8.214383       4.679882   
    50%        21.921135      10.059654       8.214383       4.679882   
    75%        24.462827      10.059654       8.722262       4.679882   
    max        26.745255      15.126244       9.051961       4.679882   
    
                      f4             f5             f6             f7  \
    count  500000.000000  500000.000000  500000.000000  500000.000000   
    mean       10.338543       4.029626      -4.154710       5.097958   
    std         0.343004       0.430263       4.573856       1.196681   
    min        10.280525      -7.191826     -27.404532       4.833815   
    25%        10.280525       4.115453      -6.699321       4.833815   
    50%        10.280525       4.115453      -2.411115       4.833815   
    75%        10.280525       4.115453       0.294443       4.833815   
    max        19.741068       4.115453       0.294443      11.998256   
    
                      f8             f9            f10            f11  
    count  500000.000000  500000.000000  500000.000000  500000.000000  
    mean        3.933725      16.016406       5.333179      -0.170974  
    std         0.056500       6.989706       0.167569       0.022783  
    min         3.644043      13.190056       5.300375      -1.219352  
    25%         3.910792      13.190056       5.300375      -0.168679  
    50%         3.971858      13.190056       5.300375      -0.168679  
    75%         3.971858      13.190056       5.300375      -0.168679  
    max         3.971858      66.714495       6.473900      -0.168679  
    
    
              f0         f1        f2        f3         f4        f5         f6  \
    0  18.700558  10.059654  8.214383  0.113022  10.280525  4.115453 -10.527786   
    1  24.892174  10.059654  8.214383  4.679882  10.280525  4.115453  -1.288207   
    2  26.162121  10.059654  8.214383  4.679882  10.280525  4.115453  -3.993764   
    3  26.180196  10.059654  8.214383  4.679882  10.280525  4.115453  -7.822229   
    4  23.779841  10.059654  8.214383  4.679882  10.280525  4.115453  -1.288207   
    
             f7        f8         f9       f10       f11  
    0  4.833815  3.971858  13.190056  5.300375 -0.168679  
    1  4.833815  3.971858  13.190056  5.300375 -0.168679  
    2  4.833815  3.971858  13.190056  5.300375 -0.168679  
    3  4.833815  3.971858  13.190056  5.300375 -0.168679  
    4  4.833815  3.971858  13.190056  5.300375 -0.168679  
    


```python
# Response and Predictors
x = train.drop(columns=['treatment', 'visit'])
y = train['treatment']

# Splitting the data into test and training set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Scaling the data
scaler = preprocessing.StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)

x_test_scaled = scaler.transform(x_test)

# Fitting Logistic Regression
model = LogisticRegression(random_state=0)
res = model.fit(x_train_scaled, y_train)
```


```python
# Compute AUC
auc = roc_auc_score(y_test, model.decision_function(x_test_scaled))

print('AUC Score: {:.3f}'.format(auc))
```

    AUC Score: 0.508
    

The model has an AUC of 0.506, indicating that it performs no better than random chance at distinguishing between the classes in the dataset. Thus, the features are independent of assignment.

Using the training data, I created two datasets:

* one for users who received the treatment
* one for users in the control group

For each dataset, I built a logistic regression model that predicts the likelihood of a website visit.  

I followed the "train/test" paradigm that we discussed in class.  

For each model, I computed three metrics using your test set:

* baseline accuracy
* accuracy
* AUC  

---
Note: the "test" set in this problem is different from the test set described in the introduction section.  

In this problem I performed the below steps:
* split the training data into two datasets: one for treatment and one for control.
* for each of these, split the data into a training set and a test set.  The training set is what you use to build the model.  The test set is what you use to calculate the evaluation metrics.


```python
# Write your code for question #2 here

# Treatment Group
treatment = train[train['treatment'] == 1]

# Control Group
control = train[train['treatment'] == 0]
```


```python
# Splitting the treatment group into training and testing sets
x_treatment = treatment.drop(columns=['treatment', 'visit'])
y_treatment = treatment['visit']
x_train_treatment, x_test_treatment, y_train_treatment, y_test_treatment = train_test_split(x_treatment, y_treatment, test_size=0.3)

# Scaling the features
scaler_treatment = preprocessing.StandardScaler().fit(x_train_treatment)
x_train_treatment_scaled = scaler_treatment.transform(x_train_treatment)

x_test_treatment_scaled = scaler_treatment.transform(x_test_treatment)

# Splitting the control group into training and testing sets
x_control = control.drop(columns=['treatment', 'visit'])
y_control = control['visit']
x_train_control, x_test_control, y_train_control, y_test_control = train_test_split(x_control, y_control, test_size=0.3)

# Scaling the features
scaler_control = preprocessing.StandardScaler().fit(x_train_control)
x_train_control_scaled = scaler_control.transform(x_train_control)

x_test_control_scaled = scaler_control.transform(x_test_control)


```


```python
# Calculate metrics for a dataset

def fit(x_train, y_train):

    # Training logistic regression model
    model = LogisticRegression(random_state=0)
    model.fit(x_train, y_train)

    return model
```


```python
# Calculate metrics for a dataset

def evaluate(y_train, x_test, y_test, model):

    # Baseline accuracy
    baseline_acc = (1 - y_train.mean())

    # Model accuracy
    model_accuracy = model.score(x_test, y_test)

    # AUC
    auc = roc_auc_score(y_test, model.decision_function(x_test))

    return baseline_acc, model_accuracy, auc
```


```python
# Creating the models

treatment_model = fit(x_train_treatment_scaled, y_train_treatment)
control_model = fit(x_train_control_scaled, y_train_control)

# Evaluation for the treatment group
baseline_acc_treatment, model_acc_treatment, auc_treatment = evaluate(y_train_treatment, x_test_treatment_scaled, y_test_treatment, treatment_model)

# Evaluation for the control group
baseline_acc_control, model_acc_control, auc_control = evaluate(y_train_control, x_test_control_scaled, y_test_control, control_model)

print("Treatment Group - Baseline Accuracy: {:.2f}, Model Accuracy: {:.2f}, AUC: {:.2f}".format(baseline_acc_treatment, model_acc_treatment, auc_treatment))
print("Control Group - Baseline Accuracy: {:.2f}, Model Accuracy: {:.2f}, AUC: {:.2f}".format(baseline_acc_control, model_acc_control, auc_control))
```

    Treatment Group - Baseline Accuracy: 0.95, Model Accuracy: 0.96, AUC: 0.93
    Control Group - Baseline Accuracy: 0.96, Model Accuracy: 0.96, AUC: 0.93
    

Both the treatment and control group models show very strong performance metrics, which indicate effective predictions for website visits in these subsets.

**Baseline Accuracy:**

The baseline accuracy is extremely high for both groups (0.95 for the treatment group and 0.96 for the control group). This metric represents the accuracy that would be achieved by always predicting the most common class. The high baseline accuracy suggests that the datasets are imbalanced, with a majority of instances likely belonging to one class (either visiting or not visiting).

**Model Accuracy:**

For both the treatment and control groups, the model accuracy almost matches the baseline accuracy (0.96 for each). This indicates that the logistic regression models are performing as well as simply predicting the most frequent outcome for each group.

**AUC (Area Under the ROC Curve):**

The AUC values are very high for both groups (0.93 for each), suggesting that the models do an excellent job at distinguishing between the visitors and non-visitors. An AUC of 0.93 indicates that there is a 93% chance that the model will rank a randomly chosen positive instance (a visitor) higher than a randomly chosen negative instance (a non-visitor).

Using the models above, I wil try to classify each of the users into one of the four groups that we discussed in class:
* Persuadables
* Sure Things
* Lost Causes
* Do Not Disturbs (aka Sleeping Dogs)

Specifically: I will calculate an "uplift score" for each user, which is the difference of the score from the treatment model and the control model. Using the score, I will create rules that will classify users into one of the four categories, and report how many users from the training data are in each category.


```python
# Write your code for question #3 here

train_test = train.drop(columns=['treatment', 'visit'])

#  Calculate the probability of visiting the website for all users
treatment_probs = treatment_model.predict_proba(scaler_treatment.transform(train_test))[:, 1]
control_probs = control_model.predict_proba(scaler_control.transform(train_test))[:, 1]

# Calculate the uplift score
uplift_score = treatment_probs - control_probs


result_dict = {
    'treatment_probs': treatment_probs,
    'control_probs': control_probs,
    'uplift_score': uplift_score
}

# Convert the dictionary to a DataFrame
result_df = pd.DataFrame(result_dict)

# Display the first few rows of the uplift scores
result_df.head()
```





  <div id="df-4138d09b-6be0-4d06-9775-7d50596d0f36" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>treatment_probs</th>
      <th>control_probs</th>
      <th>uplift_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.005441</td>
      <td>0.004563</td>
      <td>0.000879</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.023051</td>
      <td>0.017562</td>
      <td>0.005489</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.008773</td>
      <td>0.006971</td>
      <td>0.001802</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.823716</td>
      <td>0.780976</td>
      <td>0.042739</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.012865</td>
      <td>0.007655</td>
      <td>0.005210</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-4138d09b-6be0-4d06-9775-7d50596d0f36')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-4138d09b-6be0-4d06-9775-7d50596d0f36 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-4138d09b-6be0-4d06-9775-7d50596d0f36');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-d1a84bb3-81f7-42b6-a755-2fd5f4dd8ba5">
  <button class="colab-df-quickchart" onclick="quickchart('df-d1a84bb3-81f7-42b6-a755-2fd5f4dd8ba5')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-d1a84bb3-81f7-42b6-a755-2fd5f4dd8ba5 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
result_df.describe()
```





  <div id="df-5e8455ab-a7af-4b2f-8dea-bebe66206ddd" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>treatment_probs</th>
      <th>control_probs</th>
      <th>uplift_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
      <td>1000000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.047926</td>
      <td>0.040854</td>
      <td>0.007072</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.125230</td>
      <td>0.114304</td>
      <td>0.022883</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.001303</td>
      <td>0.000734</td>
      <td>-0.141362</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.006129</td>
      <td>0.004994</td>
      <td>0.001058</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.008195</td>
      <td>0.006401</td>
      <td>0.001621</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.017176</td>
      <td>0.012243</td>
      <td>0.003687</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.985771</td>
      <td>0.989229</td>
      <td>0.358525</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-5e8455ab-a7af-4b2f-8dea-bebe66206ddd')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-5e8455ab-a7af-4b2f-8dea-bebe66206ddd button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-5e8455ab-a7af-4b2f-8dea-bebe66206ddd');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-dbaac7aa-67dd-44ac-b2ba-8c0027ff3636">
  <button class="colab-df-quickchart" onclick="quickchart('df-dbaac7aa-67dd-44ac-b2ba-8c0027ff3636')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-dbaac7aa-67dd-44ac-b2ba-8c0027ff3636 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Segment the users based on the uplift score

def categorize_user(row):
    if row['uplift_score'] >= 0.0030:
      return 'Persuadables'
    elif row['uplift_score'] >= 0.0010 and row['uplift_score'] < 0.0030:
      if row['treatment_probs'] > 0.0150 and row['control_probs'] > 0.010:
        return 'Sure Things'
      if row['treatment_probs'] <= 0.0080 and row['control_probs'] <= 0.0060:
        return 'Lost Causes'
    else:
        return 'Sleeping Dogs'

```


```python
# Apply the categorization function to each row
result_df['segment'] = result_df.apply(categorize_user, axis=1)

# Count the number of users in each category
result_df['segment'].value_counts()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>segment</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Persuadables</th>
      <td>294397</td>
    </tr>
    <tr>
      <th>Lost Causes</th>
      <td>242001</td>
    </tr>
    <tr>
      <th>Sleeping Dogs</th>
      <td>206124</td>
    </tr>
    <tr>
      <th>Sure Things</th>
      <td>23074</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>



To assign users to the segments (Persuadables, Sure Things, Lost Causes, and Sleeping Dogs), these rules can be applied based on their uplift score:

1. Persuadables: Treatment = Yes, Control = No (uplift score >= 0030)
2. Sure Things: Treatment = Yes, Control = Yes (uplift score >= 0.0010 and < 0.0030, but high probability in both ie. treatment_probs > 0.0150 and control_probs > 0.010)
3. Lost Causes: Treatment = No, Control = No (uplift score >= 0.0010 and < 0.0030, but low probability in both ie. treatment_probs <= 0.0080 and control_probs <= 0.0060)
4. Sleeping Dogs: Treatment = No, Control = Yes (The rest of the cases)

The distribution of users based on segments in the training dataset is:

*   **Persuadables**: 288108
*   **Sleeping Dogs**: 241201
*   **Lost Causes**: 196098
*   **Sure Things**: 23813

Below I applied the models to the test set. Classified all users from the test set into the different uplift categories, and report the number of users in each category.


```python
# Test Set

test_set = pd.read_csv("https://github.com/hangingbelay/spring2024/raw/main/homework_4_test_data.csv.gz")

feature_columns = test_set.columns
test_set.describe()
```





  <div id="df-69f71c73-0d81-4869-86bf-459bcad0819b" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>f0</th>
      <th>f1</th>
      <th>f2</th>
      <th>f3</th>
      <th>f4</th>
      <th>f5</th>
      <th>f6</th>
      <th>f7</th>
      <th>f8</th>
      <th>f9</th>
      <th>f10</th>
      <th>f11</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>500000.000000</td>
      <td>500000.000000</td>
      <td>500000.000000</td>
      <td>500000.000000</td>
      <td>500000.000000</td>
      <td>500000.000000</td>
      <td>500000.000000</td>
      <td>500000.000000</td>
      <td>500000.000000</td>
      <td>500000.000000</td>
      <td>500000.000000</td>
      <td>500000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>19.624292</td>
      <td>10.069864</td>
      <td>8.446247</td>
      <td>4.178380</td>
      <td>10.338543</td>
      <td>4.029626</td>
      <td>-4.154710</td>
      <td>5.097958</td>
      <td>3.933725</td>
      <td>16.016406</td>
      <td>5.333179</td>
      <td>-0.170974</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.378992</td>
      <td>0.104810</td>
      <td>0.299191</td>
      <td>1.337044</td>
      <td>0.343004</td>
      <td>0.430263</td>
      <td>4.573856</td>
      <td>1.196681</td>
      <td>0.056500</td>
      <td>6.989706</td>
      <td>0.167569</td>
      <td>0.022783</td>
    </tr>
    <tr>
      <th>min</th>
      <td>12.616365</td>
      <td>10.059654</td>
      <td>8.214383</td>
      <td>-6.361766</td>
      <td>10.280525</td>
      <td>-7.191826</td>
      <td>-27.404532</td>
      <td>4.833815</td>
      <td>3.644043</td>
      <td>13.190056</td>
      <td>5.300375</td>
      <td>-1.219352</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>12.616365</td>
      <td>10.059654</td>
      <td>8.214383</td>
      <td>4.679882</td>
      <td>10.280525</td>
      <td>4.115453</td>
      <td>-6.699321</td>
      <td>4.833815</td>
      <td>3.910792</td>
      <td>13.190056</td>
      <td>5.300375</td>
      <td>-0.168679</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>21.921135</td>
      <td>10.059654</td>
      <td>8.214383</td>
      <td>4.679882</td>
      <td>10.280525</td>
      <td>4.115453</td>
      <td>-2.411115</td>
      <td>4.833815</td>
      <td>3.971858</td>
      <td>13.190056</td>
      <td>5.300375</td>
      <td>-0.168679</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>24.462827</td>
      <td>10.059654</td>
      <td>8.722262</td>
      <td>4.679882</td>
      <td>10.280525</td>
      <td>4.115453</td>
      <td>0.294443</td>
      <td>4.833815</td>
      <td>3.971858</td>
      <td>13.190056</td>
      <td>5.300375</td>
      <td>-0.168679</td>
    </tr>
    <tr>
      <th>max</th>
      <td>26.745255</td>
      <td>15.126244</td>
      <td>9.051961</td>
      <td>4.679882</td>
      <td>19.741068</td>
      <td>4.115453</td>
      <td>0.294443</td>
      <td>11.998256</td>
      <td>3.971858</td>
      <td>66.714495</td>
      <td>6.473900</td>
      <td>-0.168679</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-69f71c73-0d81-4869-86bf-459bcad0819b')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-69f71c73-0d81-4869-86bf-459bcad0819b button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-69f71c73-0d81-4869-86bf-459bcad0819b');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-9eba3583-30dc-4500-bec8-596e6d57287a">
  <button class="colab-df-quickchart" onclick="quickchart('df-9eba3583-30dc-4500-bec8-596e6d57287a')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-9eba3583-30dc-4500-bec8-596e6d57287a button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Write your code for question #4 here

treatment_probs = treatment_model.predict_proba(scaler_treatment.transform(test_set))[:, 1]

control_probs = control_model.predict_proba(scaler_control.transform(test_set))[:, 1]

uplift_score = treatment_probs - control_probs
```


```python
final_dict = {
    'treatment_probs': treatment_probs,
    'control_probs': control_probs,
    'uplift_score': uplift_score
}

# Convert the dictionary to a DataFrame
final_df = pd.DataFrame(final_dict)
final_df.head()
```





  <div id="df-a20d968e-cc98-41d7-a041-e679eb6da741" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>treatment_probs</th>
      <th>control_probs</th>
      <th>uplift_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.013259</td>
      <td>0.006675</td>
      <td>0.006583</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.005481</td>
      <td>0.004628</td>
      <td>0.000853</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.005870</td>
      <td>0.004853</td>
      <td>0.001017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.007308</td>
      <td>0.005828</td>
      <td>0.001480</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.005916</td>
      <td>0.004976</td>
      <td>0.000940</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-a20d968e-cc98-41d7-a041-e679eb6da741')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-a20d968e-cc98-41d7-a041-e679eb6da741 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-a20d968e-cc98-41d7-a041-e679eb6da741');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-5d8fac45-0f74-4af0-a19d-fe488ea2fed0">
  <button class="colab-df-quickchart" onclick="quickchart('df-5d8fac45-0f74-4af0-a19d-fe488ea2fed0')"
            title="Suggest charts"
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
      --bg-color: #E8F0FE;
      --fill-color: #1967D2;
      --hover-bg-color: #E2EBFA;
      --hover-fill-color: #174EA6;
      --disabled-fill-color: #AAA;
      --disabled-bg-color: #DDD;
  }

  [theme=dark] .colab-df-quickchart {
      --bg-color: #3B4455;
      --fill-color: #D2E3FC;
      --hover-bg-color: #434B5C;
      --hover-fill-color: #FFFFFF;
      --disabled-bg-color: #3B4455;
      --disabled-fill-color: #666;
  }

  .colab-df-quickchart {
    background-color: var(--bg-color);
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: var(--fill-color);
    height: 32px;
    padding: 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: var(--hover-bg-color);
    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: var(--button-hover-fill-color);
  }

  .colab-df-quickchart-complete:disabled,
  .colab-df-quickchart-complete:disabled:hover {
    background-color: var(--disabled-bg-color);
    fill: var(--disabled-fill-color);
    box-shadow: none;
  }

  .colab-df-spinner {
    border: 2px solid var(--fill-color);
    border-color: transparent;
    border-bottom-color: var(--fill-color);
    animation:
      spin 1s steps(1) infinite;
  }

  @keyframes spin {
    0% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
      border-left-color: var(--fill-color);
    }
    20% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    30% {
      border-color: transparent;
      border-left-color: var(--fill-color);
      border-top-color: var(--fill-color);
      border-right-color: var(--fill-color);
    }
    40% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-top-color: var(--fill-color);
    }
    60% {
      border-color: transparent;
      border-right-color: var(--fill-color);
    }
    80% {
      border-color: transparent;
      border-right-color: var(--fill-color);
      border-bottom-color: var(--fill-color);
    }
    90% {
      border-color: transparent;
      border-bottom-color: var(--fill-color);
    }
  }
</style>

  <script>
    async function quickchart(key) {
      const quickchartButtonEl =
        document.querySelector('#' + key + ' button');
      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.
      quickchartButtonEl.classList.add('colab-df-spinner');
      try {
        const charts = await google.colab.kernel.invokeFunction(
            'suggestCharts', [key], {});
      } catch (error) {
        console.error('Error during call to suggestCharts:', error);
      }
      quickchartButtonEl.classList.remove('colab-df-spinner');
      quickchartButtonEl.classList.add('colab-df-quickchart-complete');
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-5d8fac45-0f74-4af0-a19d-fe488ea2fed0 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>

    </div>
  </div>





```python
# Define the uplift categories

final_df['segment'] = final_df.apply(categorize_user, axis=1)

# Count the number of users in each category
category_counts = final_df['segment'].value_counts()

# Display the counts
category_counts
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>segment</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Persuadables</th>
      <td>146980</td>
    </tr>
    <tr>
      <th>Lost Causes</th>
      <td>121155</td>
    </tr>
    <tr>
      <th>Sleeping Dogs</th>
      <td>103379</td>
    </tr>
    <tr>
      <th>Sure Things</th>
      <td>11438</td>
    </tr>
  </tbody>
</table>
</div><br><label><b>dtype:</b> int64</label>



The distribution of users based on segments in the test dataset is:

*   **Persuadables**: 143582
*   **Sleeping Dogs**: 120734
*   **Lost Causes**: 98439
*   **Sure Things**: 11947
