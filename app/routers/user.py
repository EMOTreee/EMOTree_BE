from fastapi import APIRouter

from app.routers.dependencies import CurrentUserDep

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/me")
def read_me(current_user : CurrentUserDep):
    return current_user