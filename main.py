from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np, sklearn
from sklearn.svm import SVR

app = FastAPI()

class Req(BaseModel):
    X_train: List[List[float]]
    y_train: List[float]
    X_test: List[List[float]]
    X_last: List[float]
    params: Dict[str, Any] = {}

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(r: Req):
    p = r.params
    model = SVR(kernel=p.get("kernel", "rbf"),
                C=float(p.get("C", 100)),
                epsilon=float(p.get("epsilon", 0.01)))
    model.fit(np.array(r.X_train), np.array(r.y_train))
    return {
        "pred_test": model.predict(np.array(r.X_test)).tolist(),
        "pred_last": float(model.predict(np.array([r.X_last]))[0]),
        "sklearn_version": sklearn.__version__,
    }