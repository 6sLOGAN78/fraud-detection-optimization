"""Pipeline script to execute Part 16 — Enterprise Security, Compliance & Disaster Recovery."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.utils import (
    AuditLogger,
    ComplianceChecker,
    DisasterRecoveryManager,
    EncryptionEngine,
    IdentityAccessManager,
    PIIMasker,
    SecretsManager,
    SecurityPreExecutionGate,
    ThreatModelEngine,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Part 16 Security & Governance Pipeline")
    args = parser.parse_args()

    gate = SecurityPreExecutionGate()
    gate.verify()

    logger.info("Executing Identity & Access Management (IAM) authentication...")
    iam = IdentityAccessManager()
    role = iam.authenticate_key("key_admin_secret_123")
    auth_res = iam.authorize_permission(role or "GUEST", "deploy")

    logger.info("Executing Secrets Management & Log Masking...")
    secrets_mgr = SecretsManager()
    masked_key = secrets_mgr.mask_secret("key_admin_secret_123")

    logger.info("Executing Field Encryption Engine...")
    enc_engine = EncryptionEngine()
    cipher = enc_engine.encrypt_str("sensitive_user_ssn_12345")
    decrypted = enc_engine.decrypt_str(cipher)

    logger.info("Executing Security Audit Logging...")
    audit_logger = AuditLogger()
    audit_event = audit_logger.log_event(user_id="admin_user_01", action="DEPLOY_MODEL", resource="v1/model.joblib")

    logger.info("Executing Compliance & Governance Checks (PCI-DSS / SOC2 / GDPR)...")
    compliance = ComplianceChecker()
    comp_res = compliance.evaluate_compliance(encryption_enabled=True, audit_logging_enabled=True, rbac_enabled=True)

    logger.info("Executing Threat Model Risk Assessment...")
    threat_engine = ThreatModelEngine()
    threat_res = threat_engine.evaluate_threat_risks()

    logger.info("Executing PII Anonymization & Masking...")
    pii_masker = PIIMasker()
    df_dummy = pd.DataFrame({"email": ["john.doe@example.com", "jane@company.org"], "card": [123456, 789012]})
    df_masked = pii_masker.mask_dataframe(df_dummy, pii_columns=["email"])

    logger.info("Executing Disaster Recovery Snapshot Backup...")
    dr_mgr = DisasterRecoveryManager()
    dr_res = dr_mgr.create_snapshot_backup(source_dir="artifacts/deployment", snapshot_tag="v1_backup_snapshot")

    summary_report = {
        "iam_authentication": {"role": role, "authorized": auth_res},
        "secrets_masking": masked_key,
        "encryption": {"cipher_preview": cipher[:15] + "...", "decryption_verified": decrypted == "sensitive_user_ssn_12345"},
        "audit_event": audit_event,
        "compliance": comp_res,
        "threat_assessment": threat_res,
        "disaster_recovery": dr_res,
    }

    out_file = Path("reports/security/security_governance_summary.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary_report, f, indent=2)

    logger.info(f"Part 16 Security & Governance Pipeline completed. Report saved to {out_file}")


if __name__ == "__main__":
    main()
