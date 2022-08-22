from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import numpy as np

import rtvc_main

app = FastAPI()

class UserInput(BaseModel):
    embed: list
    text: str

@app.get('/')
def index():
    return {"Message": "This is Index"}

@app.get('/inference/')
async def inference(userinput: UserInput):
    userinput = userinput.dict()
    embed = np.array(userinput["embed"])
    text = userinput["text"]
    spec = rtvc_main.inference(embed, text)
    spec = jsonable_encoder(spec.tolist())
    return JSONResponse(spec)