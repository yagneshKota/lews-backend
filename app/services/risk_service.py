from uuid import UUID

from sqlalchemy.orm import Session

from app.models.risk_prediction import RiskPrediction
from app.ml.predictor import predictor
from app.repositories.alert_repository import AlertRepository
from app.repositories.report_repository import ReportRepository
from app.repositories.risk_repository import RiskRepository
from app.services.alert_service import AlertService
from app.services.feature_service import FeatureService
from app.websocket.manager import manager
from app.core.config import settings


class RiskService:
    def __init__(self, db: Session):
        self.db = db

        self.model = predictor
        self.report_repo = ReportRepository(db)
        self.risk_repo = RiskRepository(db)

        self.alert_service = AlertService(
            AlertRepository(db)
        )

    def predict(
        self,
        report_id: UUID,
        features: dict[str,float | int],
    ):
        # 1. Find the report
        report = self.report_repo.get(report_id)

        if report is None:
            raise ValueError("Report not found")

        # 2. Prepare the ML feature dictionary
        prepared_features = FeatureService.prepare(features)

        # 3. Run the trained ML model
        result = self.model.predict(prepared_features)

        # 4. Create database prediction
        prediction = RiskPrediction(
            report_id=report_id,
            risk_score=result.risk_score,
            risk_level=result.risk_level,
            risk_tier=result.risk_tier,
            alert_triggered=result.alert_triggered,
            alert_message=result.alert_message,
            model_version=settings.ml_model_version,
            features=features,
        )

        # 5. Save prediction
        prediction = self.risk_repo.create(prediction)

        # 6. Create alert + broadcast if required
        self.alert_service.maybe_create_and_broadcast(
            report=report,
            prediction=prediction,
            result=result,
        )

        return prediction

    def latest(self, report_id: UUID):
        return self.risk_repo.latest_for_report(report_id)