from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib

# Load model and vectorizer
model = joblib.load('modelo_fake_news.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = FastAPI(
    title="Fake News Detector API",
    description="API para detectar se uma notícia é falsa ou verdadeira",
    version="1.0.0"
)

# Templates
templates = Jinja2Templates(directory="templates")

class NewsItem(BaseModel):
    text: str


@app.post("/predict")
def predict(news: NewsItem):
    X_tfidf = vectorizer.transform([news.text])
    prediction = model.predict(X_tfidf)[0]
    probability = model.predict_proba(X_tfidf).max()
    return {
        "prediction": prediction,
        "probability": float(probability)
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# forms endpoint
@app.post("/check", response_class=HTMLResponse)
def check_news(request: Request, text: str = Form(...)):
    X_tfidf = vectorizer.transform([text])
    prediction = model.predict(X_tfidf)[0]
    probability = model.predict_proba(X_tfidf).max()
    result = f"This news was classified as **{prediction.upper()}** with {probability:.2%} of accuracy."
    return templates.TemplateResponse("index.html", {"request": request, "result": result})
