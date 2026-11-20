import streamlit as st
import pandas as pd
from transformers import pipeline

MODEL_ID = "Mahganj123/gwadarvoice-xlmr-complaints"

st.set_page_config(page_title="GwadarVoice", page_icon="🌊", layout="centered")

@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return pipeline("text-classification", model=MODEL_ID, top_k=None)

ICONS = {"Water": "💧", "Electricity": "⚡", "Roads": "🛣️", "Garbage": "🗑️", "Healthcare": "🏥", "Security": "🛡️"}
DEPT = {"Water": "Water & Sanitation Authority", "Electricity": "Power Distribution Office",
        "Roads": "Works & Communications Dept.", "Garbage": "Municipal Sanitation",
        "Healthcare": "District Health Office", "Security": "Police / Security Control Room"}
EXAMPLES = ["Marine Drive par 3 din se bijli nahi hai",
            "hospital mein doctor nahi hain aur dawai bhi nahi",
            "no drinking water in Sur Bandar since yesterday",
            "gali mein kachra jama hai, badboo aa rahi hai",
            "raat ko mobile snatching ho rahi hai"]

st.title("🌊 GwadarVoice")
st.caption("Write a civic complaint in Roman Urdu, English, or a mix. The AI routes it to the right department.")

clf = load_model()
choice = st.selectbox("Try an example (optional)", [""] + EXAMPLES)
text = st.text_area("Your complaint", value=choice, height=120, placeholder="e.g. Pishukan mein pani nahi aa raha...")

if st.button("Classify", type="primary"):
    if not text.strip():
        st.warning("Please type a complaint first.")
    else:
        res = sorted(clf(text)[0], key=lambda x: -x["score"])
        top = res[0]
        st.success(f"{ICONS.get(top['label'], '')} **{top['label']}**  -  confidence {top['score']*100:.1f}%")
        st.info(f"Suggested department: **{DEPT.get(top['label'], '-')}**")
        if top["score"] < 0.6:
            st.warning("Low confidence - the complaint may be unclear or mixed.")
        st.bar_chart(pd.DataFrame(res).set_index("label")["score"])

st.divider()
st.caption("Fine-tuned xlm-roberta-base | Educational project")
