from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Document(Base):
    """One uploaded company document. Its chunks and vectors live in ChromaDB."""
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    original_filename: Mapped[str] = mapped_column(String(255))
    stored_filename: Mapped[str] = mapped_column(String(100), unique=True)
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # SHA-256, blocks duplicates
    file_size: Mapped[int]
    chunk_count: Mapped[int] = mapped_column(default=0)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class IncidentLog(Base):
    """One uploaded application log. The raw file stays on disk, named after this id."""
    __tablename__ = "incident_logs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # the same value as log_id
    original_filename: Mapped[str] = mapped_column(String(255))
    total_entries: Mapped[int]
    error_count: Mapped[int] = mapped_column(default=0)
    uploaded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    investigations: Mapped[list["Investigation"]] = relationship(
        back_populates="log", cascade="all, delete-orphan"
    )


class Investigation(Base):
    """One agent investigation: the question, the report and the tools it used."""
    __tablename__ = "investigations"

    id: Mapped[int] = mapped_column(primary_key=True)
    log_id: Mapped[str] = mapped_column(ForeignKey("incident_logs.id"), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    steps: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(20), default="draft")  # later: approved by a human
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    log: Mapped["IncidentLog"] = relationship(back_populates="investigations")