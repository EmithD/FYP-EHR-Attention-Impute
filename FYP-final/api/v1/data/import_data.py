from fastapi import File, UploadFile, HTTPException
import pandas as pd
import io

async def import_data(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    except HTTPException:
        return {"error": "Incorrect file uploaded."}
    except:
        return {"error": "An error occured at import_data"}