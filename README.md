# 🌊 GwadarVoice - Civic Complaint Classifier

Classifies Roman Urdu / English citizen complaints (Water, Electricity, Roads, Garbage, Healthcare, Security).

- **Pretrained model:** [xlm-roberta-base](https://huggingface.co/xlm-roberta-base)
- **Fine-tuned model:** https://huggingface.co/Mahganj123/gwadarvoice-xlmr-complaints
- **Live app:** https://YOUR-APP-NAME.streamlit.app

## Dataset
Custom dataset created by the author: template-generated code-mixed complaints (Gwadar locations) + hand-written
samples. Test sets use held-out issue phrases and a hand-written set to avoid optimistic scores. See `data/`.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Pipeline
1. Dataset creation & cleaning
2. Fine-tune xlm-roberta-base (4 epochs, lr 2e-5)
3. Push to Hugging Face
4. Streamlit app loads the model from the Hub
5. Deploy on Streamlit Community Cloud
