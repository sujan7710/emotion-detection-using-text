# Emotion Detection Using Text

A machine learning Streamlit app that predicts the emotion expressed in user-entered text.

The project trains a text classifier with `scikit-learn` using combined word and character TF-IDF features with a Complement Naive Bayes classifier. The trained pipeline is saved with `joblib` and loaded by the Streamlit app for prediction.

## Emotions

- Anger
- Fear
- Joy
- Love
- Sadness
- Surprise

## Project Structure

```text
.
|-- app.py
|-- train_model.py
|-- requirements.txt
|-- data/
|   `-- emotion_dataset_expanded.csv
|-- models/
|   `-- .gitkeep
`-- README.md
```

## Run Locally

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Train the model:

```powershell
py train_model.py
```

Start Streamlit:

```powershell
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this project to GitHub.
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Create a new app from your GitHub repository.
4. Set the main file path to `app.py`.
5. Deploy.

Streamlit Cloud will install packages from `requirements.txt`. If `models/emotion_model.joblib` is missing, the app can train it automatically from `data/emotion_dataset_expanded.csv` on startup.

## Dataset

The included CSV is a balanced starter dataset for demonstration and project submission. For better accuracy, replace it with a larger emotion dataset using the same columns:

```text
text,emotion
```
