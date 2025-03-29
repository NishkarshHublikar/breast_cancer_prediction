# Breast Cancer Prediction using Machine Learning

## Overview

This project aims to predict the diagnosis of breast cancer (malignant or benign) using machine learning techniques applied to the Wisconsin Breast Cancer Diagnostic dataset. The project involves data loading, exploration, preprocessing, model training, evaluation, and visualization.

## Dataset

The Wisconsin Breast Cancer Diagnostic dataset is used in this project. It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features describe characteristics of the cell nuclei present in the image.

**Dataset Source:** The dataset can be downloaded from Kaggle or UCI Machine Learning Repository.

**Dataset Description:**

* **Features:** 30 numeric, predictive attributes are included.
* **Target Variable:** Diagnosis (M = malignant, B = benign)

## Project Workflow

1. **Data Loading:** Load the dataset from a CSV file ("data.csv") into a pandas DataFrame.
2. **Data Exploration:** Explore the DataFrame to understand its structure, identify potential issues, and prepare for data cleaning and preparation. This includes checking data types, missing values, and descriptive statistics.
3. **Data Preparation:** Prepare the data for model training by handling categorical variables, imputing missing values, and potentially scaling or normalizing features. In this project, the 'diagnosis' column is converted to numerical values (M=1, B=0).
4. **Model Training:** Train a machine learning model (Linear Regression in this case) on the prepared data.
5. **Model Evaluation:** Evaluate the model's performance using appropriate metrics such as Mean Squared Error (MSE) and R-squared.
6. **Visualization:** Visualize the model's predictions against actual values using a scatter plot.

## Code Structure

* **data.csv:** The dataset file.
* **Breast_Cancer_Wisconsin_(Diagnostic)Predict_whether_the_cancer_is_benign_or_malignant.ipynb:** Jupyter Notebook containing the code for data loading, exploration, preprocessing, model training, evaluation, and visualization.

## Dependencies

The following libraries are used in this project:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn (sklearn)

You can install them using pip:

## Usage

1. Clone this repository:
2. Download the "data.csv" dataset and place it in the repository's root directory.
3. Open the "breast_cancer_prediction.ipynb" notebook in Google Colab or Jupyter Notebook.
4. Run the notebook cells sequentially to execute the code.

## Results

The project demonstrates the application of machine learning for breast cancer prediction. The model's performance is evaluated using MSE and R-squared. The visualization provides insights into the model's predictions compared to actual values.

## Contributing

Contributions to this project are welcome. Feel free to open issues or pull requests for bug fixes, enhancements, or new features.

## License

This project is licensed under the MIT License.
