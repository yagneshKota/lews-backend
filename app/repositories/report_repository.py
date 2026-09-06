from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.report import Report
from app.schemas.report import ReportCreate


class ReportRepository:

    def __init__(self, db: Session):
        self.db = db

    def create(self, data: ReportCreate) -> Report:
        report = Report(
            longitude=data.longitude,
            latitude=data.latitude,
            report=data.report,
            report_description=data.report_description,
        )

        try:
            self.db.add(report)
            self.db.commit()
            self.db.refresh(report)
            return report

        except Exception:
            self.db.rollback()
            raise

    def get(self, report_id: UUID) -> Report | None:
        return self.db.scalar(
            select(Report).where(Report.id == report_id)
        )

    def update(self, report: Report) -> Report:
        try:
            self.db.add(report)
            self.db.commit()
            self.db.refresh(report)
            return report

        except Exception:
            self.db.rollback()
            raise

    def list(self, offset: int, limit: int) -> list[Report]:
        return list(
            self.db.scalars(
                select(Report)
                .order_by(Report.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
        )

    def count(self) -> int:
        return self.db.scalar(select(func.count()).select_from(Report)) or 0

    def delete(self, report_id: UUID) -> bool:
        report = self.get(report_id)
        if not report:
            return False
        try:
            from app.models.risk_prediction import RiskPrediction
            self.db.execute(
                select(RiskPrediction).where(RiskPrediction.report_id == report_id)
            )
            # Delete predictions associated with this report
            predictions = self.db.scalars(
                select(RiskPrediction).where(RiskPrediction.report_id == report_id)
            ).all()
            for pred in predictions:
                self.db.delete(pred)

            self.db.delete(report)
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            raise