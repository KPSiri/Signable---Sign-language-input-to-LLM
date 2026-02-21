# Signable

**Signable** is an accessibility app that allows users to provide **sign language input** to Large Language Models (LLMs). The app supports multiple pre-trained models for different vocabulary sizes and integrates with LLMs such as Claude.

---

## Features

* Real-time sign language recognition
* Supports multiple pre-trained models for various vocabularies
* Easy integration with LLMs

---

## Installation

1. **Paste your Claude API key** into `signable.py`.

2. **Create a virtual environment**:

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Run the app**:

```bash
streamlit run signable.py
```

---

## Models

Update the paths in `signable.py` according to the model you want to use:

```python
model = load_model("app/model/sign_language_recognition_new_40.keras")
scaler = joblib.load("app/model/scaler_new_40.pkl")
label_encoder = joblib.load("app/model/label_encoder_new_40.pkl")
```

Available models:

| Model Name       | Description                |
| ---------------- | -------------------------- |
| `xxnamexx`       | Original 11-word model     |
| `xxnamexx_new`   | Trained on 20 useful words |
| `xxnewxx_new_40` | Trained on 40 useful words |

---

## Notes

* Ensure the **model, scaler, and label encoder paths** in `signable.py` match the model you choose.
* Only include the models you need in deployment to reduce app size.

Do you want me to do that?
