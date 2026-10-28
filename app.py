"""Lab 04 - Phish or Legit? (Streamlit UI). Run: streamlit run app.py"""
import re
import streamlit as st
import pandas as pd

from src.pipeline import load_data, split, make_pipeline, predict, word_weights, contributions

st.set_page_config(page_title="Phish or Legit?", page_icon="🎣")
st.title("Phish or Legit?")
st.caption("TF-IDF + logistic regression on a SYNTHETIC dataset (fictional brands). "
           "Labels are 'phishing-style' vs 'legit', not real-world ground truth.")

# --- Notice obligatoire ---
st.info("⚠️ **This tool supports a human decision and does not block messages by itself. The data is synthetic.**")

@st.cache_resource
def get_model():
    df = load_data()
    Xtr, _, ytr, _ = split(df)
    # On entraîne le modèle LR et sa version calibrée
    model = make_pipeline("lr").fit(Xtr, ytr)
    model_cal = make_pipeline("lr_cal").fit(Xtr, ytr)
    return model, model_cal

model, model_cal = get_model()

# --- Exemples prédéfinis (2 phishing, 2 légitimes, dont 1 mal classé) ---
# Remplace ces textes par de vrais exemples de ton dataset, dont un que le modèle rate.
EXAMPLES = {
    "Phishing 1": "Northbank: your account has been locked. Verify now http://bit.ly/ab12cd",
    "Phishing 2": "URGENT! You have won a £1000 gift card. Click here to claim: http://fake.link/win",
    "Legit 1": "Hi team, the meeting is moved to 3pm tomorrow. Please confirm your attendance.",
    "Legit 2 (misclassified)": "Your package has been delivered. Thank you for shopping with us."  # à vérifier
}

st.subheader("1. Choose an example or type your own message")
example_choice = st.selectbox("Load an example:", ["-- Custom --"] + list(EXAMPLES.keys()))
if example_choice != "-- Custom --":
    default_text = EXAMPLES[example_choice]
else:
    default_text = "Northbank: your account has been locked. Verify now http://bit.ly/ab12cd"

text = st.text_area("Paste a message:", default_text, height=100)

# --- Slider de seuil ---
st.subheader("2. Decision threshold")
threshold = st.slider("Phishing if score ≥ threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if st.button("Classify", type="primary"):
    try:
        # On récupère le score brut (probabilité) via predict_proba
        # On suppose que model est un pipeline sklearn avec predict_proba
        proba = model.predict_proba([text])[0, 1]  # probabilité de la classe positive (phishing)
        cal_proba = model_cal.predict_proba([text])[0, 1]
        
        # Application du seuil
        label = "phish" if proba >= threshold else "legit"
        
    except ValueError as err:
        st.error(str(err))
    else:
        if label == "phish":
            st.error(f"⚠️ Looks like PHISHING-style text  ·  score {proba:.2f} (seuil {threshold:.2f})")
        else:
            st.success(f"✅ Looks legit  ·  score for phish {proba:.2f} (seuil {threshold:.2f})")
        
        st.caption(f"Calibrated score (sigmoid calibration on training data): {cal_proba:.2f}. "
                   "Scores are only meaningful as probabilities if calibration was checked - see the lab report.")

        # --- Explication par feature ---
        st.subheader("3. Feature contributions")
        st.markdown("**Words pushing toward PHISHING (red) or LEGIT (green):**")
        
        weights = word_weights(model, text)
        parts = []
        for tok in re.findall(r"\w+|\W+", text):
            w = weights.get(tok.lower(), 0.0)
            if tok.strip() and re.match(r"\w+", tok) and abs(w) > 0.3:
                parts.append(f":red[**{tok}**]" if w > 0 else f":green[**{tok}**]")
            else:
                parts.append(tok)
        st.markdown("".join(parts))
        st.caption("red = pushes toward phishing, green = pushes toward legit (|weight| > 0.3)")

        # --- Tableau détaillé des contributions ---
        contribs = contributions(model, text)
        # Trier par contribution décroissante
        sorted_contribs = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
        
        # Séparer positifs et négatifs
        pos = [(f, v) for f, v in sorted_contribs if v > 0][:5]
        neg = [(f, v) for f, v in sorted_contribs if v < 0][:5]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top features increasing score (phishing):**")
            if pos:
                df_pos = pd.DataFrame(pos, columns=["feature", "contribution to log-odds"])
                st.dataframe(df_pos, hide_index=True)
            else:
                st.write("None")
        with col2:
            st.markdown("**Top features decreasing score (legit):**")
            if neg:
                df_neg = pd.DataFrame(neg, columns=["feature", "contribution to log-odds"])
                st.dataframe(df_neg, hide_index=True)
            else:
                st.write("None")