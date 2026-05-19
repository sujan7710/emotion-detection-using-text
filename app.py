import joblib
import pandas as pd
import streamlit as st

from train_model import MODEL_PATH, train_and_save_model


EMOTION_COLORS = {
    "anger": "#b91c1c",
    "fear": "#7c3aed",
    "joy": "#b7791f",
    "love": "#be185d",
    "sadness": "#2563eb",
    "surprise": "#c2410c",
}

st.set_page_config(
    page_title="Emotion Detection Using Text",
    page_icon=":speech_balloon:",
    layout="wide",
)


@st.cache_resource(show_spinner="Training model...")
def load_model():
    if not MODEL_PATH.exists():
        return train_and_save_model()
    return joblib.load(MODEL_PATH)


def predict_emotion(text: str):
    model = load_model()
    prediction = model.predict([text])[0]

    probabilities = None
    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probability_values = model.predict_proba([text])[0]
        classes = model.named_steps["classifier"].classes_
        probabilities = dict(sorted(zip(classes, probability_values), key=lambda item: item[1], reverse=True))

    return prediction, probabilities


st.title("Emotion Detection Using Text")
st.caption("Machine learning text classifier for emotion prediction")

left_column, right_column = st.columns([1.15, 0.85], gap="large")

with left_column:
    user_text = st.text_area(
        "Enter text",
        height=210,
        placeholder="Type a message, review, social media post, or journal note...",
    )

    analyze_clicked = st.button("Predict Emotion", type="primary", use_container_width=True)

with right_column:
    st.subheader("Prediction")

    if analyze_clicked and user_text.strip():
        emotion, probabilities = predict_emotion(user_text.strip())
        emotion_label = emotion.title()
        color = EMOTION_COLORS.get(emotion, "#0f766e")

        st.markdown(
            f"""
            <div style="border:1px solid #d9e0e7;border-radius:8px;padding:22px;background:#ffffff">
                <div style="font-size:14px;color:#64707d;font-weight:700;text-transform:uppercase">Detected emotion</div>
                <div style="font-size:42px;font-weight:900;color:{color};line-height:1.1;margin-top:8px">{emotion_label}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if probabilities:
            st.write("Confidence scores")
            score_data = pd.DataFrame(
                {
                    "Emotion": [emotion.title() for emotion in probabilities.keys()],
                    "Confidence": [round(value * 100, 2) for value in probabilities.values()],
                }
            )
            st.bar_chart(score_data, x="Emotion", y="Confidence", color="#0f766e")
            st.dataframe(score_data, hide_index=True, use_container_width=True)
    elif analyze_clicked:
        st.warning("Please enter some text first.")
    else:
        st.info("Enter text and click Predict Emotion.")
