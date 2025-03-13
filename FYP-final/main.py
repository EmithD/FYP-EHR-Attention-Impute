from fastapi import FastAPI
from routes.import_data_route import router as import_data_route # type: ignore

app = FastAPI()

app.include_router(import_data_route, prefix="/api/v1/data")

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI"}