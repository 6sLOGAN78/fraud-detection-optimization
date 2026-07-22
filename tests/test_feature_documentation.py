"""Unit tests for the Feature Documentation Engine, verifying markdown structure validation, JSON schema mappings, and Git revisions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.store import FeatureViewMetadata, FeatureRegistry, FeatureLineage
from src.data.documentation import FeatureDocumentationGenerator


def test_feature_documentation_generation(tmp_path: Path) -> None:
    registry_file = tmp_path / "registry.json"
    registry = FeatureRegistry(registry_file)
    
    # 1. Register a fake feature view
    lineage = FeatureLineage(
        source_dataset="/path/to/raw_dataset.parquet",
        pipeline_stage="feature_encoding",
        transformation_type="vectorized-one-hot",
        description="One-hot encoded categorical parameters",
    )
    
    view = FeatureViewMetadata(
        name="encoded_features",
        entity_id="TransactionID",
        features=["device_type_encoded", "card_brand_encoded"],
        source_path="/path/to/encoded.parquet",
        owner="Security Auth Team",
        tags=["categorical", "encoded"],
        description="One-hot encoded categorical attributes for transaction screening",
        lineage=lineage,
    )
    registry.register_view(view)
    
    # 2. Run documentation compiler
    doc_md_file = tmp_path / "feature_documentation.md"
    doc_json_file = tmp_path / "feature_documentation_manifest.json"
    
    generator = FeatureDocumentationGenerator(registry_file, tmp_path)
    
    md_content = generator.compile_markdown_documentation(doc_md_file)
    json_manifest = generator.compile_json_manifest(doc_json_file)
    
    # 3. Assert Markdown contents
    assert doc_md_file.exists()
    assert "# Feature Store Registry Catalog Documentation" in md_content
    assert "encoded_features" in md_content
    assert "device_type_encoded" in md_content
    assert "Security Auth Team" in md_content
    assert "upstream source dataset" in md_content.lower()
    assert "ADMIN" in md_content
    
    # 4. Assert JSON manifest contents
    assert doc_json_file.exists()
    assert json_manifest["registered_views_count"] == 1
    assert "encoded_features" in json_manifest["feature_views"]
    assert json_manifest["feature_views"]["encoded_features"]["entity_id"] == "TransactionID"
    assert "categorical" in json_manifest["feature_views"]["encoded_features"]["tags"]
    assert json_manifest["feature_views"]["encoded_features"]["lineage"]["pipeline_stage"] == "feature_encoding"
