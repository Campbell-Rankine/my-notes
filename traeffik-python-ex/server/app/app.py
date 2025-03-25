from fastapi import FastAPI, HTTPException
import time
from pydantic import BaseModel


class Schema(BaseModel):
    uuid: str
    x1: float
    x2: float


app = FastAPI()


@app.get("/")
def home(data: Schema):
    return {"statusCode": 200, "msg": f"Found multiply home: {data.x1}, {data.x2}"}


@app.get("/add")
def multiply(data: Schema):
    # Simulate a program running
    time.sleep(5)
    result = data.x1 + data.x2  # addition for easy verification
    return {"statusCode": 200, "msg": f"Program Finished. Calculation: {result}"}


@app.get("/multiply")
def multiply(data: Schema):
    # Simulate a program running
    time.sleep(5)
    result = data.x1 * data.x2  # addition for easy verification
    return {"statusCode": 200, "msg": f"Program Finished. Calculation: {result}"}
