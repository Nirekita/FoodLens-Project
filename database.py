from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timedelta
import json, os

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# Render gives postgres:// but SQLAlchemy needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

engine       = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base         = declarative_base()

class CachedProduct(Base):
    __tablename__ = "products"
    barcode      = Column(String, primary_key=True)
    product_name = Column(String)
    result_json  = Column(Text)        # full scored result, JSON string
    cached_at    = Column(DateTime, default=datetime.utcnow)

class ScanLog(Base):
    __tablename__ = "scans"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    barcode      = Column(String)
    product_name = Column(String)
    ifhi_score   = Column(Float)
    deception    = Column(Float)
    scanned_at   = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

CACHE_TTL_DAYS = 7

def get_cached(barcode: str):
    db = SessionLocal()
    try:
        row = db.query(CachedProduct).filter_by(barcode=barcode).first()
        if not row:
            return None
        age = datetime.utcnow() - row.cached_at
        if age > timedelta(days=CACHE_TTL_DAYS):
            db.delete(row)
            db.commit()
            return None
        return json.loads(row.result_json)
    finally:
        db.close()

def save_cache(barcode: str, product_name: str, result: dict):
    db = SessionLocal()
    try:
        row = CachedProduct(
            barcode      = barcode,
            product_name = product_name,
            result_json  = json.dumps(result),
            cached_at    = datetime.utcnow(),
        )
        db.merge(row)
        db.commit()
    finally:
        db.close()

def log_scan(barcode: str, product_name: str, ifhi: float, deception: float):
    db = SessionLocal()
    try:
        db.add(ScanLog(
            barcode      = barcode,
            product_name = product_name,
            ifhi_score   = ifhi,
            deception    = deception,
        ))
        db.commit()
    finally:
        db.close()