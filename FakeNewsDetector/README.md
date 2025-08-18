# Fake News Detector 📰❌✅

This project detects whether news articles are **fake** or **true** using Machine Learning.  
It provides a **FastAPI** backend to make predictions on new texts.

---


### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the virtual environment


```bash
venv\Scripts\Activate
```

### 3. Install Dependencies


```bash
pip install -r requirements.txt
```


### 4. Train the model
```bash
python model_train.py
```
### 5. Run API


```bash
uvicorn main:app --reload
