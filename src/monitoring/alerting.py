"""12.8 Alerting Framework Engine Module.

Provides threshold monitoring alerts, severity rules (INFO, WARNING, CRITICAL),
and notification logging for drift and performance breaches.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertingEngine:
    """12.8 Manages alert triggers, severity classifications, and alert logging."""

    def __init__(self, alert_log_file: str = "logs/monitoring/alerts.json"):
        self.alert_log_file = Path(alert_log_file)
        self.alert_log_file.parent.mkdir(parents=True, exist_ok=True)
        self.active_alerts: List[Dict[str, Any]] = []

    def trigger_alert(
        self,
        alert_name: str,
        message: str,
        severity: str = "WARNING",
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Triggers an alert event and logs to active alert store and file."""
        alert_event = {
            "alert_name": alert_name,
            "message": message,
            "severity": severity.upper(),
            "metrics": metrics or {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.active_alerts.append(alert_event)

        if severity.upper() == "CRITICAL":
            logger.critical(f"CRITICAL ALERT [{alert_name}]: {message}")
        elif severity.upper() == "WARNING":
            logger.warning(f"WARNING ALERT [{alert_name}]: {message}")
        else:
            logger.info(f"INFO ALERT [{alert_name}]: {message}")

        self._save_alerts()
        return alert_event

    def _save_alerts(self) -> None:
        """Persists alerts to JSON log file."""
        try:
            with open(self.alert_log_file, "w") as f:
                json.dump(self.active_alerts, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist alerts log: {e}")

    def evaluate_drift_and_alert(self, drift_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluates drift monitoring output and fires alerts automatically."""
        triggered = []
        if drift_results.get("has_severe_data_drift", False):
            evt = self.trigger_alert(
                alert_name="DataDriftDetected",
                message=f"Severe data drift detected in {drift_results.get('drifted_features_count', 0)} features.",
                severity="WARNING",
                metrics=drift_results,
            )
            triggered.append(evt)
        return triggered
