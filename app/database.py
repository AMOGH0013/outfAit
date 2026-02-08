import os
from sqlalchemy import event
from sqlalchemy import text
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./wardrobe.db",
)

engine_kwargs = {"echo": False}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)

if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, connection_record):  # noqa: ARG001
        # Some Windows/dev setups throw "disk I/O error" when SQLite tries to use
        # rollback journals/WAL files in the project directory. Disabling journaling
        # keeps the hackathon MVP working reliably.
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=OFF")
            cursor.execute("PRAGMA synchronous=OFF")
        finally:
            cursor.close()

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    pass


def ensure_sqlite_schema():
    """
    Lightweight SQLite migrations for hackathon/MVP use.

    SQLAlchemy's `create_all()` won't add new columns to existing tables.
    This function keeps the local `wardrobe.db` in sync with model additions.
    """
    if not str(engine.url).startswith("sqlite"):
        return

    with engine.begin() as conn:
        # Add columns to wardrobe_items if missing
        try:
            rows = conn.execute(text("PRAGMA table_info(wardrobe_items)")).fetchall()
        except Exception:
            return

        existing_cols = {row[1] for row in rows}  # row[1] = column name
        if "color_palette" not in existing_cols:
            conn.execute(text("ALTER TABLE wardrobe_items ADD COLUMN color_palette TEXT"))
        if "embedding" not in existing_cols:
            conn.execute(text("ALTER TABLE wardrobe_items ADD COLUMN embedding TEXT"))
        if "suggested_item_type" not in existing_cols:
            conn.execute(text("ALTER TABLE wardrobe_items ADD COLUMN suggested_item_type TEXT"))
        if "suggested_item_type_confidence" not in existing_cols:
            conn.execute(text("ALTER TABLE wardrobe_items ADD COLUMN suggested_item_type_confidence REAL"))

        # Backfill columns for body_profiles if table exists (create_all won't add columns later).
        try:
            bp_rows = conn.execute(text("PRAGMA table_info(body_profiles)")).fetchall()
        except Exception:
            bp_rows = []
        if bp_rows:
            bp_cols = {row[1] for row in bp_rows}
            if "height_cm" not in bp_cols:
                conn.execute(text("ALTER TABLE body_profiles ADD COLUMN height_cm INTEGER"))
            if "weight_kg" not in bp_cols:
                conn.execute(text("ALTER TABLE body_profiles ADD COLUMN weight_kg INTEGER"))
            if "user_name" not in bp_cols:
                conn.execute(text("ALTER TABLE body_profiles ADD COLUMN user_name TEXT"))
            if "sex" not in bp_cols:
                conn.execute(text("ALTER TABLE body_profiles ADD COLUMN sex TEXT"))
            if "age" not in bp_cols:
                conn.execute(text("ALTER TABLE body_profiles ADD COLUMN age INTEGER"))
            if "skin_tone" not in bp_cols:
                conn.execute(text("ALTER TABLE body_profiles ADD COLUMN skin_tone TEXT"))


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
