import uuid
from pathlib import Path
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.repositories.report_repository import ReportRepository
from app.schemas.report import PaginatedReportsResponse, ReportCreate, ReportResponse
from app.services.report_service import ReportService


router = APIRouter(prefix="/api/reports", tags=["reports"])


UPLOAD_DIR = Path("uploads/reports")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


@router.post("", response_model=ReportResponse, status_code=201)
def create_report(
    payload: ReportCreate,
    db: Session = Depends(get_db),
):
    return ReportService(db).create(payload)


@router.get("", response_model=PaginatedReportsResponse)
def get_reports(
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    return ReportService(db).list(
        offset=offset,
        limit=limit,
    )


@router.get("/{report_id}", response_model=ReportResponse)
def get_report(
    report_id: UUID,
    db: Session = Depends(get_db),
):
    report = ReportService(db).get(report_id)

    if report is None:
        raise HTTPException(
            status_code=404,
            detail="Report not found",
        )
    return report

@router.delete("/{report_id}", status_code=204)
def delete_report(
    report_id: UUID,
    db: Session = Depends(get_db),
):
    ReportService(db).delete(report_id)
    return None


@router.post("/{report_id}/image")
async def upload_report_image(
    report_id: UUID,
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    repository = ReportRepository(db)

    # Check that the report exists
    report = repository.get(report_id)

    if report is None:
        raise HTTPException(
            status_code=404,
            detail="Report not found",
        )

    # Validate image type
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG, PNG and WEBP images are allowed",
        )

    # Read image
    contents = await image.read()

    # Validate file size
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="Image must be smaller than 10 MB",
        )

    # Generate safe filename
    extension = ALLOWED_TYPES[image.content_type]
    filename = f"{uuid.uuid4()}{extension}"

    # Save image
    file_path = UPLOAD_DIR / filename

    with open(file_path, "wb") as file:
        file.write(contents)

    # Store image URL in PostgreSQL
    report.image_url = f"/uploads/reports/{filename}"

    repository.update(report)

    return {
        "message": "Image uploaded successfully",
        "image_url": report.image_url,
    }