# Customer Churn Prediction using Artificial Neural Network (ANN)

## Overview

This project demonstrates how to predict customer churn using a deep learning-based Artificial Neural Network (ANN). It includes feature engineering, data preprocessing, model building, and deploying a real-time prediction app with Streamlit.

## Project Highlights

- **Feature Engineering:** Encoded categorical variables into numerical features to improve model performance.
- **Data Preprocessing:** Applied feature scaling and data splitting (training and testing sets).
- **Efficient Storage:** Saved the encoder and scaler using pickle files for easy reuse and deployment.
- **Deep Learning Model:** Trained an ANN model to predict whether customers will churn.
- **Streamlit Web App:** Built a user-friendly web interface for making real-time predictions.
- **Deployment:** Deployed the app on Streamlit Cloud for online access.


## Installation

To set up the project locally, clone the repository and install the required dependencies:

## Clone the repository:
   ```bash
   git clone https://github.com/your-username/customer-churn-prediction-ann.git
   cd customer-churn-prediction-ann
   ```
## Install the necessary Python packages:
```bash
pip install -r requirements.txt
```
## Dataset
The dataset consists of customer details, including:

Demographic information (age, gender, etc.)
Account details (tenure, monthly charges, etc.)
Usage statistics (service calls, complaints, etc.)

## How to Run
## Build the model using the model.ipynb notebook:

```bash
jupyter notebook notebooks/model.ipynb
```

## Make predictions on new customer data using the prediction.ipynb notebook:
```bash
jupyter notebook notebooks/prediction.ipynb
```

## Run the Streamlit app for real-time predictions:
```bash
streamlit run app.py
```

## Model Evaluation
The model’s performance is evaluated based on Accuracy

## App Deployment
The app is deployed on Streamlit Cloud for public access. You can use the app to predict customer churn by simply uploading customer data.

## Future Improvements
- Experimenting with advanced deep learning architectures.
- Hyperparameter tuning for better performance.
- Adding more features to enhance the model's prediction capabilities.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Check it out
🔗 [Live App - Customer Churn Prediction](https://tiaenapwmgjgpasayv2pyf.streamlit.app/)



