# Customer Churn Prediction System

A beginner-friendly Machine Learning project that predicts whether a customer is likely to leave a service provider. The project demonstrates the basic Machine Learning lifecycle and how to serve predictions through **FastAPI** and display them in an interactive **Streamlit** application.

> **Learning level:** Beginner  
> **Main language:** Python  
> **Machine Learning model:** Logistic Regression (classification)  
> **API framework:** FastAPI  
> **User interface:** Streamlit

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Problem](#2-business-problem)
3. [Project Objectives](#3-project-objectives)
4. [How the Application Works](#4-how-the-application-works)
5. [Project Architecture](#5-project-architecture)
6. [Project Folder Structure](#6-project-folder-structure)
7. [Technology Stack](#7-technology-stack)
8. [Machine Learning Concepts](#8-machine-learning-concepts)
9. [Role of Each Python File](#9-role-of-each-python-file)
10. [Installation and Setup](#10-installation-and-setup)
11. [How to Run the Project](#11-how-to-run-the-project)
12. [Example Prediction Flow](#12-example-prediction-flow)
13. [API Overview](#13-api-overview)
14. [Model Evaluation](#14-model-evaluation)
15. [Beginner Learning Roadmap](#15-beginner-learning-roadmap)
16. [Common Issues](#16-common-issues)
17. [Possible Improvements](#17-possible-improvements)
18. [Important Notes](#18-important-notes)

---

## 1. Project Overview

Customer Churn Prediction is a supervised Machine Learning project. It uses historical customer information to learn patterns associated with customers who leave a company.

The trained model can make a prediction for a new customer. A Streamlit interface collects the customer's details, sends them to a FastAPI backend, and displays the result returned by the model.

For example, a business may want to identify customers who could stop using its telecom service. The business can use the prediction to decide which customers may need follow-up or additional support.

**Important:** The prediction is an estimate based on the data used to train the model. It is not a guarantee that a customer will leave.

## 2. Business Problem

Companies may lose customers because of pricing, service quality, contract terms, or other reasons. Losing customers can reduce revenue and increase the cost of acquiring new customers.

The business question is:

**“Based on the available customer information, is this customer likely to churn?”**

The project treats this as a **binary classification** problem:

- `1` or `Yes`: the customer churned / is classified as likely to churn, depending on the dataset's label definition.
- `0` or `No`: the customer did not churn / is classified as unlikely to churn, depending on the dataset's label definition.

The meaning of the target label must be checked in the dataset before training the model.

## 3. Project Objectives

By completing this project, a beginner can learn to:

- Read and explore a CSV dataset.
- Understand features (input columns) and the target (the value to predict).
- Identify and handle missing or inconsistent data.
- Convert categorical values into a format the model can use.
- Split data into training and testing sets.
- Train a Logistic Regression model.
- Evaluate classification performance using suitable metrics.
- Save a trained model and reuse it for new predictions.
- Write reusable Python functions.
- Create a REST API with FastAPI.
- Create an interactive user interface with Streamlit.
- Connect a frontend application to a backend API.

## 4. How the Application Works

The project has three main stages.

### Stage A: Train the model

1. Load the historical customer dataset.
2. Inspect the columns, data types, missing values, and target labels.
3. Separate input features (`X`) from the target (`y`).
4. Split the data into training and testing sets.
5. Preprocess numeric and categorical features as required.
6. Train the Logistic Regression model using the training data.
7. Evaluate it using the held-out test data.
8. Save the complete fitted preprocessing-and-model pipeline.

Training is performed separately from making predictions. The application should load the saved model rather than retrain it every time a user requests a prediction.

### Stage B: Serve predictions through FastAPI

1. Start the FastAPI server.
2. Load the saved model pipeline.
3. Receive customer details through an API endpoint.
4. Validate the incoming values.
5. Pass the values to the model.
6. Return a prediction and, if implemented, a probability estimate as JSON.

### Stage C: Display results in Streamlit

1. The user opens the Streamlit application.
2. A form asks for the customer information required by the model.
3. The user submits the form.
4. Streamlit sends the details to the FastAPI endpoint.
5. FastAPI runs the model prediction and returns a response.
6. Streamlit displays the result to the user.

## 5. Project Architecture

```text
                    MODEL TRAINING
                    ---------------
                     customers.csv
                           |
                           v
                   train_model.py
               (clean, prepare, train,
                  evaluate and save)
                           |
                           v
                    Saved ML Pipeline
                           |
                           | loaded by
                           v
USER --> Streamlit UI --> FastAPI --> ML Pipeline
          app.py          api.py         |
              ^                           |
              |                           v
              +------ Prediction JSON <---+
```

### What does each component mean?

| Component | Responsibility |
|---|---|
| Dataset | Provides historical examples for learning |
| `train_model.py` | Prepares data, trains, evaluates, and saves the model |
| Saved model pipeline | Stores the fitted preprocessing steps and trained model |
| `api.py` | Receives requests and returns model predictions |
| `app.py` | Collects user input and displays the API response |
| User | Enters customer details and reviews the prediction |

**Key idea:** Streamlit is the presentation layer, FastAPI is the API/backend layer, and the saved Machine Learning pipeline makes the prediction.

## 6. Project Folder Structure

The following is a suggested beginner-friendly structure. File names can be adjusted to match the implementation.

```text
customer-churn-prediction/
│
├── data/
│   └── customers.csv
│
├── model/
│   └── churn_pipeline.pkl
│
├── download_data.py
├── train_model.py
├── api.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Folder and file explanations

| File or folder | Purpose |
|---|---|
| `data/` | Stores the dataset used for training, subject to the dataset's licence and privacy requirements |
| `data/customers.csv` | Local copy of the IBM Telco Customer Churn sample dataset, downloaded by `download_data.py` |
| `model/` | Stores the trained model pipeline |
| `model/churn_pipeline.joblib` | Saved preprocessing-and-model pipeline; created after successful training |
| `download_data.py` | Downloads and validates the public IBM Telco Customer Churn sample dataset |\n| `train_model.py` | Contains the Python functions for loading data, preparing features, training, evaluating, and saving the pipeline |
| `api.py` | Defines the FastAPI application and prediction endpoint |
| `app.py` | Defines the Streamlit user interface |
| `requirements.txt` | Lists the Python packages required to run the project |
| `.gitignore` | Prevents virtual environments, secrets, caches, and other unneeded files from being committed |
| `README.md` | Explains the project, setup, architecture, and usage |

> The structure above is a recommended layout, not a claim that all these files already exist. The saved model file is generated by the training script. Keep the dataset filename, model path, feature names, and API endpoint consistent across the implementation.

## 7. Technology Stack

| Technology | Why it is used |
|---|---|
| Python | Main programming language |
| Pandas | Loads and explores tabular data |
| NumPy | Supports numerical operations where needed |
| Scikit-learn | Splits data, preprocesses features, trains the model, and evaluates results |
| Logistic Regression | Beginner-friendly classification algorithm |
| Joblib or Pickle | Saves and loads the fitted model pipeline |
| FastAPI | Exposes predictions through HTTP endpoints |
| Uvicorn | Runs the FastAPI application |
| Streamlit | Builds the interactive web interface |
| Requests | Allows Streamlit to call the FastAPI endpoint |

Use only one model-serialization method in the actual implementation. The example filename above uses `.pkl`, but the project may instead use Joblib and a `.joblib` extension.

## 8. Machine Learning Concepts

### 8.1 Supervised learning

Supervised learning uses historical examples that include both the input information and the known answer. The model learns a relationship between them.

In this project:

- **Inputs (`X`):** customer features such as tenure, monthly charges, contract type, and service details, depending on the dataset.
- **Target (`y`):** the historical churn label.
- **Training:** the model learns patterns from training examples.
- **Prediction:** the model estimates the target for a new customer.

### 8.2 Classification

Classification predicts a category rather than a continuous numeric quantity. This project predicts one of two churn classes, so it is a **binary classification** task.

### 8.3 Logistic Regression

Despite the word “Regression” in its name, Logistic Regression is commonly used for classification. It estimates the probability of a class and uses a decision threshold to assign a predicted label.

The default threshold is often `0.5`, but it should not automatically be treated as the best business threshold. The appropriate threshold depends on the validation results and the relative cost of false positives and false negatives.

### 8.4 Training and testing

The data is divided so that the model is evaluated on examples it did not use for fitting. This gives a more realistic estimate of performance on unseen data.

- **Training set:** used to fit the model.
- **Test set:** held back for final evaluation.
- **Validation data or cross-validation:** can help compare settings and choose a threshold without repeatedly tuning against the test set.

### 8.5 Preprocessing

Machine Learning algorithms require data in an appropriate format.

Examples include:

- Handling missing values.
- Encoding categorical values, such as contract type.
- Scaling numeric features if appropriate.

Fit preprocessing steps on the training data only. A scikit-learn `Pipeline` and `ColumnTransformer` can help ensure the same fitted transformations are applied during both evaluation and later predictions.

### 8.6 Avoiding data leakage

Data leakage occurs when information unavailable at prediction time improperly influences model training or evaluation. For example, fitting a scaler or imputer on the entire dataset before splitting can leak information from the test set.

Split the data first, fit preprocessing on the training set, and use the fitted pipeline to transform test and new customer data. Exclude features that directly reveal the outcome or would not be known when the prediction is made.

## 9. Role of Each Python File

### `train_model.py` — model training

This file should contain small, clearly named functions. For example:

- `load_data()` — reads the dataset.
- `prepare_features()` — separates features and target and prepares the preprocessing pipeline.
- `train_model()` — fits the model on the training data.
- `evaluate_model()` — calculates evaluation metrics.
- `save_model()` — saves the fitted pipeline.

These are suggested function responsibilities. The final function names should match the code.

The training script should also check that the target column exists and that the dataset contains usable rows and both classes where required.

### `api.py` — FastAPI backend

This file should:

- Create the FastAPI application.
- Load the saved pipeline.
- Define an input schema for customer details.
- Validate requests.
- Provide a health-check endpoint if desired.
- Define a prediction endpoint, such as `POST /predict`.
- Return a clear JSON response and handle invalid input or unavailable model files appropriately.

FastAPI can generate interactive API documentation automatically, commonly at `/docs`.

### `app.py` — Streamlit interface

This file should:

- Display a project title and short instructions.
- Collect the model's required input fields.
- Validate required user input.
- Send the submitted values to FastAPI.
- Show the prediction returned by the API.
- Handle connection failures gracefully.

The Streamlit app should call the API instead of loading and independently running a second copy of the model. This keeps the responsibilities clear for beginners.

### `requirements.txt` — dependencies

This file lists the Python packages needed by the project. Include the packages actually imported by the code and use compatible versions. Do not add libraries that the project does not need.

## 10. Installation and Setup

### Prerequisites

Install:

- Python (use a version compatible with the chosen package versions).
- A code editor such as Visual Studio Code.
- A terminal or command prompt.

### Step 1: Get the project

Clone your repository or open the project folder in your terminal.

```bash
git clone <YOUR-REPOSITORY-URL>
cd customer-churn-prediction
```

Replace `<YOUR-REPOSITORY-URL>` with your actual repository URL. If you already have the files locally, open that folder instead.

### Step 2: Create a virtual environment

**Windows (Command Prompt):**

```bat
python -m venv .venv
.venv\Scripts\activate
```

**macOS or Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install dependencies

After creating `requirements.txt`, run:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

A typical dependency list for this project may include:

```text
pandas
numpy
scikit-learn
joblib
fastapi
uvicorn
streamlit
requests
```

These are package names, not pinned production versions. Once the project is tested, record compatible versions in `requirements.txt` for repeatable installation.

### Step 4: Download the dataset

From the project root, run:

```bash
python download_data.py
```

This downloads the public IBM Telco Customer Churn sample dataset and validates that it contains the expected `Churn` target before saving it to `data/customers.csv`. The sample represents a fictional telecommunications provider and is intended for learning. See the source repository: https://github.com/IBM/telco-customer-churn-on-icp4d.

If you use a different dataset, place it at `data/customers.csv` and check its documentation for the target column, feature names, label values, and columns that should be excluded.

### Step 5: Train the model

Run:

```bash
python train_model.py
```

The script prints test-set evaluation results and saves the fitted pipeline to `model/churn_pipeline.joblib`.

If training fails, check the dataset path, target column, input column names, missing values, and package versions.

## 11. How to Run the Project

Run the API and Streamlit interface in **two separate terminals**. Keep the virtual environment activated in both.

### Terminal 1: Start FastAPI

```bash
uvicorn api:app --reload
```

This command assumes that `api.py` contains a FastAPI object named `app` and that you are running the command from the project root.

Open the interactive API documentation:

- `http://127.0.0.1:8000/docs`

Keep this terminal running while using Streamlit.

### Terminal 2: Start Streamlit

```bash
streamlit run app.py
```

Streamlit normally opens a browser tab. If it does not, use the local URL printed in the terminal, commonly:

- `http://localhost:8501`

### Recommended run order

1. Install dependencies.
2. Place the dataset in the correct location.
3. Run `python train_model.py` and confirm the model was saved.
4. Start FastAPI.
5. Start Streamlit.
6. Enter valid customer details and submit the form.
7. Confirm that Streamlit displays the response returned by FastAPI.

## 12. Example Prediction Flow

Imagine a user enters these illustrative details:

| Input | Example value |
|---|---|
| Tenure | 6 months |
| Monthly charges | 85.00 |
| Contract type | Month-to-month |

These are only examples. They are not guaranteed to match the columns, units, or categories in your selected dataset.

The flow is:

1. Streamlit collects the values.
2. Streamlit converts them into a JSON request with the exact feature names expected by the API.
3. FastAPI validates the request.
4. The saved pipeline applies the same preprocessing used during training.
5. The model predicts a churn class.
6. FastAPI returns a JSON response.
7. Streamlit displays the returned class.

An illustrative API response might look like this:

```json
{
  "prediction": 1,
  "label": "Likely to churn"
}
```

This is an example response format only. The actual response fields and label mapping must match the implemented API and the dataset's target definition. Do not display a probability unless the model and API actually calculate and return it.

## 13. API Overview

An API allows two software components to communicate using a defined request and response format.

### Example endpoint design

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Optional welcome or service information |
| `GET` | `/health` | Check whether the service and model are available |\n| `GET` | `/schema` | Return the feature names and category choices required by the trained model |
| `POST` | `/predict` | Accept customer features and return a prediction |

These endpoints are recommendations. They are available only if implemented in `api.py`.

### What happens during `POST /predict`?

**Request:** a JSON object containing the required customer features.

**Processing:** FastAPI validates the input and passes the data to the loaded model pipeline.

**Response:** JSON containing the predicted class and any additional values explicitly supported by the implementation.

Use the automatically generated documentation at `/docs` to inspect the actual request schema, submit test requests, and review the response.

## 14. Model Evaluation

Do not judge a churn model using accuracy alone. In an imbalanced dataset, a model may achieve high accuracy while failing to identify many customers who churn.

| Metric | Beginner-friendly explanation |
|---|---|
| Accuracy | Fraction of all predictions that are correct |
| Precision | Among customers predicted to churn, the fraction who actually churned |
| Recall | Among customers who actually churned, the fraction correctly identified |
| F1-score | A combined measure of precision and recall |
| Confusion matrix | Counts correct and incorrect predictions for each class |
| ROC-AUC | Measures how well the model ranks positive cases above negative cases across thresholds |

For customer churn, recall may be important when missing a customer who is about to leave is costly. Precision also matters if contacting customers has a significant cost. Choose metrics and a prediction threshold based on the business objective.

**Do not promise a specific score in advance.** Actual performance depends on the dataset, preprocessing, split strategy, class balance, and model configuration.

## 15. Beginner Learning Roadmap

Follow this order so learners understand each component before integrating the full application.

1. **Understand the business problem:** define churn and the target label.
2. **Explore the dataset:** inspect rows, columns, data types, and target distribution.
3. **Prepare the data:** handle missing values and encode categorical columns.
4. **Train a baseline model:** use Logistic Regression.
5. **Evaluate the model:** examine the confusion matrix, precision, recall, and F1-score.
6. **Save and reload the pipeline:** verify that a saved model can make a prediction.
7. **Build the API:** accept sample input and return a prediction.
8. **Test the API:** use `/docs` before connecting the interface.
9. **Build the Streamlit form:** collect the same features required by the API.
10. **Integrate the components:** send form submissions to FastAPI and show the result.
11. **Test common failures:** invalid inputs, missing model file, and API unavailable.
12. **Document the project:** add screenshots, dataset attribution, example requests, and real evaluation results.

## 16. Common Issues

| Problem | What to check |
|---|---|
| Dataset file not found | Confirm the file path and working directory |
| Target column error | Check the real target column name in the dataset |
| Model file not found | Run the training script successfully before starting the API |
| `uvicorn` command not found | Activate the virtual environment and install the requirements |
| Streamlit cannot connect | Confirm FastAPI is running at the URL configured in `app.py` |
| API returns validation error | Ensure the JSON fields, types, and category values match the API schema |
| Unknown category during prediction | Configure preprocessing to handle valid unseen categories appropriately |
| Predictions differ from expectations | Check feature order/names, preprocessing, target mapping, and evaluation results |
| High accuracy but poor churn detection | Review class balance, recall, precision, F1-score, and the confusion matrix |

## 17. Possible Improvements

After the basic version works, extend the project gradually:

- Compare Logistic Regression with Decision Tree and Random Forest.
- Add a confusion matrix and evaluation metrics to the dashboard.
- Show a churn probability only if the API returns a properly calculated probability.
- Add data validation and clearer error messages.
- Add unit tests for preprocessing and prediction functions.
- Add logging and API health checks.
- Store model version information and training metrics.
- Deploy the API and Streamlit app to a suitable hosting environment.
- Add privacy, access control, and monitoring before using real customer data.

Keep the initial version small. Build and test one feature at a time before adding advanced functionality.

## 18. Important Notes

- **Dataset:** Provide the source and licence of the dataset you use. Do not commit private customer information.
- **Target definition:** Verify which target value means churn before mapping predictions to labels.
- **Model file safety:** Only load model files from trusted sources. Pickle-based formats can execute code when loaded.
- **Privacy:** Use sample or properly anonymized data for learning and demonstrations.
- **Business use:** This is an educational prototype, not a guarantee of customer behaviour. Evaluate fairness, reliability, privacy, and business impact before real-world use.
- **Reproducibility:** Record package versions, dataset provenance, training settings, and evaluation results.
- **Implementation alignment:** Keep the README, actual file names, feature schema, model path, and API routes synchronized with the code.

---

## Project Summary

This project brings together three essential skills:

- **Machine Learning:** learns patterns from historical customer data.
- **FastAPI:** makes the trained model available through an API.
- **Streamlit:** provides a simple interface for users to request and view predictions.

The central lesson is the end-to-end workflow: **data → preprocessing → model training → evaluation → saved model → API → user interface → prediction**.

A successful beginner project is not one that uses the most technologies. It is one where you can explain each step, validate the model properly, and demonstrate how the parts work together.
