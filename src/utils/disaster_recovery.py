"""16.9 Disaster Recovery and Backup Engine Module.

Provides automated checkpoint snapshotting, artifact backup, and point-in-time recovery:
- 16.9 Disaster Recovery & Snapshot Backup Manager
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DisasterRecoveryManager:
    """16.9 Manages automated snapshot backups of model artifacts and point-in-time disaster recovery."""

    def __init__(self, backup_dir: str = "artifacts/backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def create_snapshot_backup(
        self,
        source_dir: Union[str, Path] = "artifacts/deployment",
        snapshot_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates a timestamped snapshot backup of production deployment artifacts."""
        src = Path(source_dir)
        if not src.exists():
            return {"backup_created": False, "reason": f"Source directory {src} does not exist."}

        tag = snapshot_tag or time.strftime("snapshot_%Y%m%d_%H%M%S")
        dest_dir = self.backup_dir / tag

        shutil.copytree(src, dest_dir, dirs_exist_ok=True)

        manifest = {
            "snapshot_tag": tag,
            "source_dir": str(src),
            "backup_path": str(dest_dir),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(dest_dir / "backup_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Disaster Recovery Snapshot created at {dest_dir}")
        return {"backup_created": True, "snapshot_tag": tag, "backup_path": str(dest_dir)}

    def restore_from_snapshot(
        self, snapshot_tag: str, target_dir: Union[str, Path] = "artifacts/deployment"
    ) -> bool:
        """Restores deployment artifacts from a specific snapshot tag."""
        backup_path = self.backup_dir / snapshot_tag
        if not backup_path.exists():
            logger.error(f"Recovery Failed: Snapshot tag '{snapshot_tag}' not found at {backup_path}")
            return False

        target = Path(target_dir)
        shutil.copytree(backup_path, target, dirs_exist_ok=True)
        logger.info(f"Disaster Recovery SUCCESS: Restored snapshot '{snapshot_tag}' into {target}")
        return True
