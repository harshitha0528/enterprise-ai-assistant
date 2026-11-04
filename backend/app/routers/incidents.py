from fastapi import APIRouter

router = APIRouter(
    prefix="/incidents",
    tags=["Incident Intelligence"]
)


@router.get("/")
def get_incidents():
    return {
        "message": "Incident Intelligence module is ready"
    }