from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from app.database import engine, Base, SessionLocal, ensure_sqlite_schema
from app.models import user, scan, wardrobe, outfit, feedback, assistant
from app.models import body_profile
from app.api import outfits
from app.api import scan as scan_api
from app.api import wardrobe as wardrobe_api
from app.api import feedback as feedback_api
from app.api import body_profile as body_profile_api
from app.api import style_profile as style_profile_api



app = FastAPI()
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
_FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
app.mount("/frontend", StaticFiles(directory=str(_FRONTEND_DIR)), name="frontend")


@app.get("/")
def serve_index():
    return FileResponse(str(_FRONTEND_DIR / "index.html"))

@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    ensure_sqlite_schema()
    db = SessionLocal()
    try:
        existing = db.query(user.User).first()
        if not existing:
            db.add(
                user.User(
                    email="dev@example.com",
                    password_hash="dev",
                )
            )
            db.commit()
    finally:
        db.close()

app.include_router(outfits.router)
app.include_router(scan_api.router)
app.include_router(wardrobe_api.router)
app.include_router(feedback_api.router)
app.include_router(body_profile_api.router)
app.include_router(style_profile_api.router)
