from uuid import UUID

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.ml.predictor import predictor
from app.repositories.report_repository import ReportRepository
from app.repositories.risk_repository import RiskRepository
from app.schemas.report import PaginatedReportsResponse, ReportCreate, ReportResponse
from app.services.risk_service import RiskService


class ReportService:
    predictor = predictor

    def __init__(self, db: Session):
        self.db = db
        self.repo = ReportRepository(db)
        self.risk_repo = RiskRepository(db)

    def create(self, payload: ReportCreate) -> ReportResponse:
        report = self.repo.create(payload)
        resp = ReportResponse.model_validate(report)

        if payload.features:
            risk_service = RiskService(self.db)
            pred = risk_service.predict(report.id, payload.features)
            resp.risk_score = pred.risk_score
            resp.risk_tier = pred.risk_tier
            resp.risk_label = pred.risk_tier

        return resp

    def get(self, report_id: UUID) -> ReportResponse:
        obj = self.repo.get(report_id)

        if obj is None:
            raise HTTPException(
                status_code=404,
                detail="Report not found",
            )

        resp = ReportResponse.model_validate(obj)
        latest_risk = self.risk_repo.latest_for_report(report_id)
        if latest_risk:
            resp.risk_score = latest_risk.risk_score
            resp.risk_tier = latest_risk.risk_tier
            resp.risk_label = latest_risk.risk_tier

        return resp

    def list(self, offset: int, limit: int) -> PaginatedReportsResponse:
        reports = self.repo.list(
            offset=offset,
            limit=limit,
        )
        total = self.repo.count()
        items: list[ReportResponse] = []
        for rep in reports:
            item = ReportResponse.model_validate(rep)
            latest_risk = self.risk_repo.latest_for_report(rep.id)
            if latest_risk:
                item.risk_score = latest_risk.risk_score
                item.risk_tier = latest_risk.risk_tier
                item.risk_label = latest_risk.risk_tier
            items.append(item)

        return PaginatedReportsResponse(
            items=items,
            total=total,
            offset=offset,
            limit=limit,
        )

    def delete(self, report_id: UUID) -> bool:
        success = self.repo.delete(report_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Report not found",
            )
        return True