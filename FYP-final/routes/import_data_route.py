from fastapi import APIRouter
from api.v1.data.import_data import import_data

router = APIRouter(tags=["TransformerModel"])

@router.get("/import-data")
def getTransformer():
    return import_data()