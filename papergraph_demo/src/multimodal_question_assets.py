from __future__ import annotations

import json
from pathlib import Path


def load_multimodal_asset_index(graph: dict, base_dir: Path) -> dict[str, dict]:
    rel_path = str(graph.get("multimodal_assets_path") or "").strip()
    if not rel_path:
        return {}
    path = base_dir / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Master graph references missing multimodal assets file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(asset.get("asset_id", "")).strip(): asset
        for asset in payload.get("assets", [])
        if isinstance(asset, dict) and str(asset.get("asset_id", "")).strip()
    }


def asset_references_for_kcs(kcs: list[dict], asset_index: dict[str, dict] | None = None) -> list[dict]:
    asset_index = asset_index or {}
    refs_by_asset: dict[str, dict] = {}
    for kc in kcs:
        if not isinstance(kc, dict) or not bool(kc.get("modality", {}).get("is_multimodal")):
            continue
        asset_id = str(kc.get("asset_id", "")).strip()
        if not asset_id:
            raise ValueError(f"Multimodal KC {kc.get('kc_id')} has no asset_id.")
        asset = asset_index.get(asset_id, {})
        ref = refs_by_asset.setdefault(
            asset_id,
            {
                "asset_id": asset_id,
                "asset_type": kc.get("asset_type") or asset.get("asset_type"),
                "caption": kc.get("asset_caption") or asset.get("caption"),
                "summary": kc.get("asset_summary") or asset.get("nearby_context"),
                "requires_multimodal_input": True,
                "input_kind": _input_kind(kc.get("asset_type") or asset.get("asset_type")),
                "target_kc_ids": [],
                "evidence_bases": [],
                "attachments": _asset_attachments(asset),
            },
        )
        kc_id = str(kc.get("kc_id", "")).strip()
        if kc_id and kc_id not in ref["target_kc_ids"]:
            ref["target_kc_ids"].append(kc_id)
        evidence_basis = str(kc.get("asset_evidence_basis") or "").strip()
        if evidence_basis and evidence_basis not in ref["evidence_bases"]:
            ref["evidence_bases"].append(evidence_basis)
    return list(refs_by_asset.values())


def attach_asset_references(
    item: dict,
    by_kc: dict[str, dict],
    asset_index: dict[str, dict] | None = None,
    target_key: str = "target_kc_ids",
) -> dict:
    refs = asset_references_for_kcs(
        [
            by_kc[kc_id]
            for kc_id in item.get(target_key, [])
            if kc_id in by_kc
        ],
        asset_index=asset_index,
    )
    item["requires_multimodal_input"] = bool(refs)
    item["asset_references"] = refs
    return item


def attach_asset_references_to_questions(
    questions: list[dict],
    by_kc: dict[str, dict],
    asset_index: dict[str, dict] | None = None,
) -> list[dict]:
    return [attach_asset_references(dict(question), by_kc, asset_index) for question in questions]


def attach_asset_references_to_challenge_plans(
    challenge_plans: dict,
    by_kc: dict[str, dict],
    asset_index: dict[str, dict] | None = None,
) -> dict:
    updated = dict(challenge_plans)
    plans = []
    for plan in challenge_plans.get("challenge_plans", []):
        item = dict(plan)
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        refs = asset_references_for_kcs(
            [
                by_kc[kc_id]
                for kc_id in source.get("kc_ids", [])
                if kc_id in by_kc
            ],
            asset_index=asset_index,
        )
        metadata = dict(item.get("metadata") or {})
        if refs:
            metadata["asset_references"] = refs
            metadata["requires_multimodal_input"] = True
        item["metadata"] = metadata
        plans.append(item)
    updated["challenge_plans"] = plans
    return updated


def _input_kind(asset_type: object) -> str:
    text = str(asset_type or "").strip().lower()
    if text == "figure":
        return "image"
    if text == "table":
        return "table"
    return text or "unknown"


def _asset_attachments(asset: dict) -> list[dict]:
    if not asset:
        return []
    asset_type = str(asset.get("asset_type") or "").strip().lower()
    if asset_type == "figure":
        attachments = asset.get("attachments")
        if isinstance(attachments, list) and attachments:
            return [
                {
                    "type": str(item.get("type") or "image"),
                    "asset_id": item.get("asset_id") or asset.get("asset_id"),
                    "path": item.get("path"),
                    "caption": item.get("caption") or asset.get("caption"),
                }
                for item in attachments
                if isinstance(item, dict) and str(item.get("path") or "").strip()
            ]
        return [
            {
                "type": "image",
                "asset_id": asset.get("asset_id"),
                "path": path,
                "caption": asset.get("caption"),
            }
            for path in asset.get("image_paths", [])
            if str(path).strip()
        ]
    if asset_type == "table":
        attachments = []
        markdown = str(asset.get("normalized_markdown") or "").strip()
        if markdown:
            attachments.append(
                {
                    "type": "table_markdown",
                    "asset_id": asset.get("asset_id"),
                    "content": markdown,
                    "caption": asset.get("caption"),
                }
            )
        return attachments
    return []
