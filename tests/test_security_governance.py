"""Unit tests for Part 16 — Enterprise Security, Compliance & Disaster Recovery."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

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


def test_16_1_to_16_4_security_framework():
    gate = SecurityPreExecutionGate()
    assert gate.verify() is True

    iam = IdentityAccessManager()
    role = iam.authenticate_key("key_admin_secret_123")
    assert role == "ADMIN"
    assert iam.authorize_permission("ADMIN", "deploy") is True

    secrets_mgr = SecretsManager()
    assert secrets_mgr.mask_secret("12345678") == "12****78"

    enc = EncryptionEngine()
    cipher = enc.encrypt_str("secret_data")
    assert enc.decrypt_str(cipher) == "secret_data"


def test_16_5_to_16_8_compliance_and_governance():
    with tempfile.TemporaryDirectory() as tmpdir:
        audit_file = Path(tmpdir) / "audit.jsonl"
        audit = AuditLogger(log_file=str(audit_file))
        evt = audit.log_event("user1", "READ", "data")
        assert evt["user_id"] == "user1"
        assert audit_file.exists()

    comp = ComplianceChecker()
    c_res = comp.evaluate_compliance()
    assert c_res["overall_compliant"] is True

    threat = ThreatModelEngine()
    t_res = threat.evaluate_threat_risks()
    assert t_res["overall_security_posture"] == "SECURE"

    pii = PIIMasker()
    assert pii.mask_email("john.doe@example.com") == "j***@example.com"
    df = pd.DataFrame({"email": ["test@domain.com"]})
    df_m = pii.mask_dataframe(df, ["email"])
    assert df_m["email"].iloc[0] == "t***@domain.com"


def test_16_9_disaster_recovery():
    with tempfile.TemporaryDirectory() as tmp_backup:
        with tempfile.TemporaryDirectory() as tmp_src:
            (Path(tmp_src) / "model.joblib").write_text("model weights")

            dr = DisasterRecoveryManager(backup_dir=tmp_backup)
            res = dr.create_snapshot_backup(source_dir=tmp_src, snapshot_tag="snap_1")
            assert res["backup_created"] is True

            with tempfile.TemporaryDirectory() as tmp_dest:
                restore_ok = dr.restore_from_snapshot("snap_1", target_dir=tmp_dest)
                assert restore_ok is True
                assert (Path(tmp_dest) / "model.joblib").exists()
