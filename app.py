from hate.pipeline.train_pipeline import TrainPipeline
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sys
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from hate.pipeline.prediction_pipeline import PredictionPipeline
from hate.exception import CustomException
from hate.constants import *


class PredictRequest(BaseModel):
    """Request body for 4-class sentiment prediction (Positive, Negative, Neutral, Irrelevant)."""
    text: str


app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")




@app.get("/train")
async def training():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    


@app.post("/predict")
async def predict_route(req: PredictRequest):
    try:
        obj = PredictionPipeline()
        label = obj.run_pipeline(req.text)
        return {"sentiment": label}
    except Exception as e:
        raise CustomException(e, sys) from e
    



if __name__=="__main__":
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)


    
# train_pipeline = TrainPipeline()

# train_pipeline.run_pipeline()