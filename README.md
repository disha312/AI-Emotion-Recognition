
# AI-Based Emotion Recognition from Text

## Project Overview
This project detects human emotions from text using Machine Learning. 
It classifies text into one of six emotions:
- Sadness
- Joy
- Love
- Anger
- Fear
- Surprise

## Dataset
The model was trained on a Kaggle Emotion Dataset containing:
- 16,000 training samples
- 2,000 validation samples
- 2,000 testing samples

## Model Used
- TF-IDF Vectorization
- Logistic Regression Classifier

## Model Performance
Accuracy on test data: ~89%

## How It Works
1. Text input is converted into numerical features using TF-IDF.
2. Logistic Regression predicts the emotion.
3. The UI displays predicted emotion with confidence score.

## Technologies Used
- Python
- Scikit-learn
- Streamlit
- Pandas

## How to Run

1. Activate virtual environment: venv\Scripts\activate

2. Run the app: streamlit run app.py