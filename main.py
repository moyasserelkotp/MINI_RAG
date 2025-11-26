from fastapi import FastAPI


app = FastAPI()


@app.get("/app_name")
async def get_app_name():
    return {"app_name": "MINI-TOURISM_RAG"}