from fastapi import APIRouter
from .endpoints import auth

# ����1
admin_router = APIRouter()

# ����1
admin_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["��-��"]
)

# �����v֡!W�1
# admin_router.include_router(users.router, prefix="/users", tags=["��-(7�"])
# admin_router.include_router(agents.router, prefix="/agents", tags=["��-��"])