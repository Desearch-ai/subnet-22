from fastapi import FastAPI

app = FastAPI(
    title="SN22 Utility API",
    description="Subnet-22 (Desearch) dataset & logging utility API",
    version="0.1.0",
)


@app.get("/")
async def root():
    return {"message": "Subnet-22 utility api is running!"}
