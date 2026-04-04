import os
import sys
import yaml
import pickle
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime

# Dynamically map the housing module so Pickle can un-serialize the CustomTransformer
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from datetime import datetime
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

Base = declarative_base()

class InferenceRecord(Base):
    __tablename__ = "inferences"
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    input_data = Column(String)
    prediction = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

engine = create_engine("sqlite:///./inferences.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI(title="Housing Price Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class InputData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

class RecordResponse(BaseModel):
    id: int
    model_name: str
    input_data: str
    prediction: float
    timestamp: datetime
    class Config:
        orm_mode = True

# Global cache for loaded models
MODELS_CACHE = {}

def get_prediction(data_dict, model_name="random_forest"):
    global MODELS_CACHE
    if model_name not in MODELS_CACHE:
        model_path = f"artifacts/{model_name}.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model pickle not found in {model_path}")
        with open(model_path, "rb") as f:
            MODELS_CACHE[model_name] = pickle.load(f)
        
    model = MODELS_CACHE[model_name]
    df = pd.DataFrame([data_dict])
    pred = model.predict(df)
    return pred[0]

@app.post("/predict/{model_name}", response_model=RecordResponse)
def predict(model_name: str, input_data: InputData, db: Session = Depends(get_db)):
    data_dict = input_data.dict()
    try:
        prediction = float(get_prediction(data_dict, model_name))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    record = InferenceRecord(
        model_name=model_name,
        input_data=str(data_dict),
        prediction=prediction
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

@app.get("/inferences/{model_name}")
def get_inferences(model_name: str, db: Session = Depends(get_db)):
    records = db.query(InferenceRecord).filter(InferenceRecord.model_name == model_name).all()
    return records

@app.get("/metrics")
def get_metrics():
    """Return per-model evaluation metrics from artifacts/metrics.txt"""
    metrics_path = "artifacts/metrics.txt"
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="Metrics file not found. Run evaluate_models.py first.")
    result = {}
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: "model_name: RMSE=x, MAE=y, R2=z"
            name, rest = line.split(":", 1)
            parts = {}
            for kv in rest.split(","):
                k, v = kv.strip().split("=")
                parts[k.strip()] = float(v.strip())
            result[name.strip()] = parts
    return result
