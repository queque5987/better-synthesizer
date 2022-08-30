from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse, FileResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import json

import rtvc_main

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    embed: list
    text: str

@app.get('/')
def index():
    return FileResponse("index.html")

@app.post('/inference/')
async def inference(userinput: UserInput):
    userinput = userinput.dict()
    embed = np.array(userinput["embed"])
    text = userinput["text"]
    spec = rtvc_main.inference(embed, text)
    # spec = jsonable_encoder(spec.tolist())
    spec = spec.tolist()
    spec_json = json.dumps({
        "spec": spec
    })
    return JSONResponse(spec_json)