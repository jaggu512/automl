import os
from datetime import datetime, timezone
from urllib.parse import quote_plus

from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(128), index=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    task_type: Mapped[str] = mapped_column(String(64))
    target_column: Mapped[str] = mapped_column(String(255))
    dataset_rows: Mapped[int] = mapped_column(Integer)
    dataset_columns: Mapped[int] = mapped_column(Integer)
    model_path: Mapped[str] = mapped_column(String(512))
    top_model_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    summary_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(128), index=True)
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    prediction_value: Mapped[str] = mapped_column(String(255))
    encrypted_input: Mapped[str] = mapped_column(String(4096))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )


def _database_url():
    direct = os.getenv("DATABASE_URL")
    if direct:
        return direct

    host = os.getenv("MYSQL_HOST", "127.0.0.1")
    port = os.getenv("MYSQL_PORT", "3306")
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    database = os.getenv("MYSQL_DATABASE", "automl")
    encoded_user = quote_plus(user)
    encoded_password = quote_plus(password)
    return f"mysql+pymysql://{encoded_user}:{encoded_password}@{host}:{port}/{database}"


ENGINE = create_engine(_database_url(), pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False)


def init_db():
    Base.metadata.create_all(bind=ENGINE)


def ping_db():
    with ENGINE.connect() as connection:
        connection.execute(text("SELECT 1"))
