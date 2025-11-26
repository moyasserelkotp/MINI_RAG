from fastapi import FastAPI, APIRouter
import os


base_router = APIRouter(
    prefix="/base",
    tags=["base_routes"],
)


@base_router.get("/app_details")
async def get_app_name():
    app_name = os.getenv("APP_NAME")
    app_version = os.getenv("APP_VERSION")
    return {"app_name": app_name, "app_version": app_version}
