from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.kc_type_registry import VALID_KC_TYPES, valid_kc_type
from src.model_client import OpenAICompatClient
from src.progress import log
from src.prompt_loader import load_prompt, render_prompt
from src.rubric_builder import build_kc_rubric


def build_kc_bank(
    paper_id: str,
    candidates: list[dict],
    macro_spine: dict,
    client: OpenAICompatClient,
    allow_offline_fallback: bool = False,
) -> dict:
    if not candidates:
        raise ValueError("Cannot build KC Bank from an empty candidate pool.")
    if not client or not client.is_ready():
        raise RuntimeError("KC Bank construction requires a configured online LLM client.")
    if not client.embeddings_ready():
        raise RuntimeError("KC Bank evidence scoring requires EMBED_MODEL and a configured embeddings endpoint.")

    macro_by_id = {
        m["macro_id"]: m
        for m in macro_spine.get("macro_nodes", [])
        if m.get("macro_id")
    }
    if not macro_by_id:
        raise ValueError("KC Bank construction requires a non-empty Macro Spine.")

    max_bank = _env_int("KC_BANK_MAX", 0, minimum=0)
    selected_candidates = candidates if max_bank <= 0 else candidates[:max_bank]
    nodes = []
    for idx, candidate in enumerate(selected_candidates, start=1):
        nodes.append(_candidate_to_node(candidate, idx, macro_by_id))

    duplicate_summary = _attach_duplicate_metadata(nodes)
    _attach_evidence_quality(nodes, client)
    _attach_llm_subjective_scores(nodes, macro_spine, client)
    _attach_rubrics(nodes, client, allow_offline_fallback)
    log(
        "KC Bank built",
        candidates=len(candidates),
        bank_kcs=len(nodes),
        semantic_merge="disabled",
        duplicate_groups=duplicate_summary["duplicate_group_count"],
        duplicate_kcs=duplicate_summary["duplicate_kc_count"],
    )
    return {"paper_id": paper_id, "kc_nodes": nodes, "duplicate_summary": duplicate_summary}


def append_kc_candidates_to_bank(
    kc_bank: dict,
    candidates: list[dict],
    macro_spine: dict,
    client: OpenAICompatClient,
    allow_offline_fallback: bool = False,
) -> dict:
    if not candidates:
        raise ValueError("Cannot append KC Bank nodes from an empty candidate pool.")
    if not client or not client.is_ready():
        raise RuntimeError("KC Bank extension requires a configured online LLM client.")
    if not client.embeddings_ready():
        raise RuntimeError("KC Bank extension requires EMBED_MODEL and a configured embeddings endpoint.")
    macro_by_id = {
        m["macro_id"]: m
        for m in macro_spine.get("macro_nodes", [])
        if m.get("macro_id")
    }
    if not macro_by_id:
        raise ValueError("KC Bank extension requires a non-empty Macro Spine.")

    existing_nodes = kc_bank.get("kc_nodes", [])
    if not isinstance(existing_nodes, list) or not existing_nodes:
        raise ValueError("KC Bank extension requires a non-empty existing kc_nodes list.")
    existing_candidate_ids = {
        str(node.get("source_candidate_id", "")).strip()
        for node in existing_nodes
        if str(node.get("source_candidate_id", "")).strip()
    }
    new_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.get("candidate_id", "")).strip() not in existing_candidate_ids
    ]
    if not new_candidates:
        raise ValueError("No new KC candidates to append; all candidate IDs already exist in KC Bank.")

    start_idx = len(existing_nodes) + 1
    new_nodes = [
        _candidate_to_node(candidate, start_idx + offset, macro_by_id)
        for offset, candidate in enumerate(new_candidates)
    ]
    _attach_evidence_quality(new_nodes, client)
    _attach_llm_subjective_scores(new_nodes, macro_spine, client)
    _attach_rubrics(new_nodes, client, allow_offline_fallback)
    kc_bank["kc_nodes"] = existing_nodes + new_nodes
    kc_bank["duplicate_summary"] = _attach_duplicate_metadata(kc_bank["kc_nodes"])
    kc_bank.pop("score_metadata", None)
    kc_bank.setdefault("extension_metadata", {})["last_appended_candidate_count"] = len(new_candidates)
    kc_bank["extension_metadata"]["final_scores_stale"] = True
    kc_bank["extension_metadata"]["stale_reason"] = "multimodal_kc_candidates_appended"
    log(
        "KC Bank extended",
        appended=len(new_nodes),
        total=len(kc_bank.get("kc_nodes", [])),
    )
    return kc_bank


def finalize_kc_bank_scores(
    kc_bank: dict,
    macro_spine: dict,
    reasoning_edges: list[dict],
) -> None:
    macro_scores = _macro_centrality_scores(macro_spine)
    graph_scores = _graph_connectivity_scores(kc_bank.get("kc_nodes", []), reasoning_edges, macro_spine)
    for node in kc_bank.get("kc_nodes", []):
        macro_centrality = macro_scores.get(node["macro_id"], 0.0)
        graph_connectivity = graph_scores.get(node["kc_id"], 0.0)
        scores = node.setdefault("importance_scores", {})
        scores["macro_centrality"] = macro_centrality
        scores["graph_connectivity"] = graph_connectivity
        _require_score(scores, "evidence_quality", node["kc_id"])
        _require_score(scores, "claim_specificity", node["kc_id"])
        _require_score(scores, "questionability", node["kc_id"])
        final_score = (
            0.30 * scores["macro_centrality"]
            + 0.25 * scores["evidence_quality"]
            + 0.20 * scores["claim_specificity"]
            + 0.15 * scores["questionability"]
            + 0.10 * scores["graph_connectivity"]
        )
        scores["final_importance_score"] = round(final_score, 4)
        node["scores"] = scores


def _candidate_to_node(candidate: dict, idx: int, macro_by_id: dict[str, dict]) -> dict:
    macro_id = str(candidate.get("macro_id", "")).strip()
    if macro_id not in macro_by_id:
        raise ValueError(f"KC candidate {candidate.get('candidate_id')} references invalid macro_id={macro_id!r}.")
    claim = str(candidate.get("claim", "")).strip()
    if not claim:
        raise ValueError(f"KC candidate {candidate.get('candidate_id')} has empty claim.")
    evidence_text = str(candidate.get("evidence", "")).strip() or claim
    evidence_items = _candidate_evidence_items(candidate, evidence_text, idx)
    kc_type = _valid_type(candidate.get("type")) or _infer_type_from_macro(macro_by_id[macro_id].get("role", ""))
    importance = _valid_importance(candidate.get("importance")) or macro_by_id[macro_id].get("importance", "normal")
    if importance not in {"critical", "normal"}:
        importance = "normal"
    return {
        "kc_id": f"KC{idx}",
        "source_candidate_id": candidate.get("candidate_id", f"C{idx}"),
        "unit_id": candidate.get("unit_id", ""),
        "source_window_id": candidate.get("source_window_id", ""),
        "macro_id": macro_id,
        "type": kc_type,
        "source_section": candidate.get("section", ""),
        "source_section_id": candidate.get("section_id", ""),
        "source_span_ids": _source_span_ids(candidate),
        "short_label": _short_label(claim),
        "claim": claim,
        "full_claim": claim,
        "evidence_text": evidence_text,
        "evidence": evidence_items,
        "claim_strength": str(candidate.get("claim_strength", "")).strip(),
        "scope": candidate.get("scope", {}),
        "related_terms": _string_list(candidate.get("related_terms", [])),
        "modality": candidate.get("modality", {"is_multimodal": False}),
        "asset_id": candidate.get("asset_id"),
        "asset_type": candidate.get("asset_type"),
        "asset_caption": candidate.get("asset_caption"),
        "asset_summary": candidate.get("asset_summary"),
        "asset_evidence_basis": candidate.get("asset_evidence_basis"),
        "asset_source_basis": candidate.get("asset_source_basis", []),
        "asset_possible_misreadings": candidate.get("asset_possible_misreadings", []),
        "asset_needs_review": candidate.get("asset_needs_review", False),
        "asset_confidence": candidate.get("asset_confidence"),
        "candidate_forbidden_claims": candidate.get("forbidden_claims", []),
        "importance": importance,
        "importance_scores": {},
        "llm_scores_raw": {},
        "flags": {
            "active_for_question_generation": False,
            "active_for_core_metrics": False,
            "usable_for_claim_verification": True,
        },
    }


def _attach_evidence_quality(nodes: list[dict], client: OpenAICompatClient) -> None:
    texts = []
    for node in nodes:
        texts.append(node["full_claim"])
        texts.append(node["evidence_text"])
    vectors = client.embed_texts(texts)
    for idx, node in enumerate(nodes):
        claim_vec = vectors[2 * idx]
        evidence_vec = vectors[2 * idx + 1]
        sim = _cosine_similarity(claim_vec, evidence_vec)
        quality = max(0.0, min(1.0, (sim - 0.45) / 0.40))
        node["importance_scores"]["evidence_quality"] = round(quality, 4)
        node["importance_scores"]["evidence_similarity"] = round(sim, 4)


def _attach_llm_subjective_scores(
    nodes: list[dict],
    macro_spine: dict,
    client: OpenAICompatClient,
) -> None:
    tpl = load_prompt("score_kc_subjective.txt")
    payload = [
        {
            "kc_id": node["kc_id"],
            "macro_id": node["macro_id"],
            "type": node["type"],
            "full_claim": node["full_claim"],
            "evidence": node["evidence_text"][:1200],
        }
        for node in nodes
    ]
    result = client.chat_json(
        system_prompt="You score subjective KC quality for paper evaluation. Return JSON only.",
        user_prompt=render_prompt(
            tpl,
            macro_spine_json=json.dumps(macro_spine, ensure_ascii=False, indent=2),
            kc_nodes_json=json.dumps(payload, ensure_ascii=False, indent=2),
        ),
        temperature=0.1,
    )
    raw_scores = result.get("scores", [])
    if not isinstance(raw_scores, list):
        raise ValueError("score_kc_subjective response must contain scores list.")
    by_id = {item.get("kc_id"): item for item in raw_scores if isinstance(item, dict)}
    missing = [node["kc_id"] for node in nodes if node["kc_id"] not in by_id]
    if missing:
        raise ValueError(f"Subjective KC scoring missing KC IDs: {missing}")

    for node in nodes:
        raw = by_id[node["kc_id"]]
        specificity_raw = _score_1_5(raw, "claim_specificity_score", node["kc_id"])
        questionability_raw = _score_1_5(raw, "questionability_score", node["kc_id"])
        node["importance_scores"]["claim_specificity"] = round((specificity_raw - 1) / 4, 4)
        node["importance_scores"]["questionability"] = round((questionability_raw - 1) / 4, 4)
        node["llm_scores_raw"] = {
            "claim_specificity_score": specificity_raw,
            "questionability_score": questionability_raw,
            "reason": str(raw.get("reason", "")).strip(),
        }


def _attach_rubrics(
    nodes: list[dict],
    client: OpenAICompatClient,
    allow_offline_fallback: bool,
) -> None:
    workers = _env_int("RUBRIC_ONLINE_WORKERS", 4, minimum=1)
    futures = {}
    failed: list[tuple[dict, Exception]] = []
    with ThreadPoolExecutor(max_workers=min(workers, len(nodes))) as ex:
        for node in nodes:
            futures[
                ex.submit(
                    _build_rubric_for_node,
                    node,
                    client,
                    allow_offline_fallback,
                )
            ] = node
        for fut in as_completed(futures):
            node = futures[fut]
            try:
                _attach_rubric_result(node, fut.result())
                log("KC Bank rubric generated", kc_id=node["kc_id"])
            except Exception as exc:
                failed.append((node, exc))
                log(
                    "KC Bank rubric generation failed; queued for sequential retry",
                    kc_id=node["kc_id"],
                    error=f"{type(exc).__name__}: {exc}",
                )
    for node, first_exc in failed:
        try:
            _attach_rubric_result(node, _build_rubric_for_node(node, client, allow_offline_fallback))
            log("KC Bank rubric generated after retry", kc_id=node["kc_id"])
        except Exception as final_exc:
            raise RuntimeError(
                f"KC Bank rubric generation failed for {node['kc_id']} after batch retry. "
                f"First error: {type(first_exc).__name__}: {first_exc}. "
                f"Final error: {type(final_exc).__name__}: {final_exc}"
            ) from final_exc


def _build_rubric_for_node(
    node: dict,
    client: OpenAICompatClient,
    allow_offline_fallback: bool,
) -> dict:
    return build_kc_rubric(
        node["kc_id"],
        node["full_claim"],
        node["evidence_text"],
        node["type"],
        node["importance"],
        client,
        allow_offline_fallback,
    )


def _attach_rubric_result(node: dict, rubric: dict) -> None:
    canonical_evidence = node.get("evidence", [])
    candidate_forbidden = node.get("candidate_forbidden_claims", [])
    rubric_evidence = rubric.pop("evidence", None)
    rubric_forbidden = rubric.pop("forbidden_claims", [])
    node.update(rubric)
    node["evidence"] = canonical_evidence
    if rubric_evidence:
        node["rubric_evidence"] = rubric_evidence
    node["forbidden_claims"] = _merge_forbidden_claims(candidate_forbidden, rubric_forbidden)


def _merge_forbidden_claims(left: object, right: object) -> list[dict]:
    merged = []
    seen = set()
    for collection in (left, right):
        if not isinstance(collection, list):
            continue
        for item in collection:
            if not isinstance(item, dict):
                continue
            key = (str(item.get("claim_id", "")).strip(), str(item.get("claim", "")).strip())
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


def _attach_duplicate_metadata(nodes: list[dict]) -> dict:
    for node in nodes:
        node["similarity_group_id"] = None
        node["near_duplicate_kc_ids"] = []
        node["dedup_status"] = "unique"
        node["duplicate_match_type"] = "none"

    parent = {node["kc_id"]: node["kc_id"] for node in nodes}
    group_has_near_match: dict[str, bool] = {}

    def find(kc_id: str) -> str:
        while parent[kc_id] != kc_id:
            parent[kc_id] = parent[parent[kc_id]]
            kc_id = parent[kc_id]
        return kc_id

    def union(left: str, right: str, near_match: bool) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            group_has_near_match[left_root] = group_has_near_match.get(left_root, False) or near_match
            return
        keep, move = sorted([left_root, right_root])
        parent[move] = keep
        group_has_near_match[keep] = (
            group_has_near_match.get(left_root, False)
            or group_has_near_match.get(right_root, False)
            or near_match
        )

    normalized_claims = {
        node["kc_id"]: _normalize_claim_for_duplicate_check(node.get("full_claim", ""))
        for node in nodes
    }
    token_sets = {
        node["kc_id"]: set(normalized_claims[node["kc_id"]].split())
        for node in nodes
    }

    exact_buckets: dict[str, list[str]] = {}
    for node in nodes:
        norm = normalized_claims[node["kc_id"]]
        if norm:
            exact_buckets.setdefault(norm, []).append(node["kc_id"])
    for bucket in exact_buckets.values():
        if len(bucket) <= 1:
            continue
        first = bucket[0]
        for kc_id in bucket[1:]:
            union(first, kc_id, near_match=False)

    threshold = _env_float("KC_NEAR_DUPLICATE_JACCARD", 0.92, minimum=0.5, maximum=1.0)
    min_tokens = _env_int("KC_NEAR_DUPLICATE_MIN_TOKENS", 5, minimum=1)
    by_id = {node["kc_id"]: node for node in nodes}
    for idx, left in enumerate(nodes):
        left_id = left["kc_id"]
        left_tokens = token_sets[left_id]
        if len(left_tokens) < min_tokens:
            continue
        for right in nodes[idx + 1 :]:
            right_id = right["kc_id"]
            if find(left_id) == find(right_id):
                continue
            if left.get("macro_id") != right.get("macro_id"):
                continue
            right_tokens = token_sets[right_id]
            if len(right_tokens) < min_tokens:
                continue
            if _token_jaccard(left_tokens, right_tokens) >= threshold:
                union(left_id, right_id, near_match=True)

    groups: dict[str, list[str]] = {}
    for node in nodes:
        groups.setdefault(find(node["kc_id"]), []).append(node["kc_id"])
    duplicate_groups = [sorted(values, key=_kc_sort_key) for values in groups.values() if len(values) > 1]

    exact_group_count = 0
    near_group_count = 0
    for idx, kc_ids in enumerate(sorted(duplicate_groups, key=lambda ids: _kc_sort_key(ids[0])), start=1):
        group_id = f"SG{idx}"
        root = find(kc_ids[0])
        has_near = group_has_near_match.get(root, False)
        if has_near:
            near_group_count += 1
            status = "preserved_near_duplicate"
            match_type = "near_exact_token_jaccard"
        else:
            exact_group_count += 1
            status = "preserved_exact_duplicate"
            match_type = "exact_normalized_claim"
        for kc_id in kc_ids:
            node = by_id[kc_id]
            node["similarity_group_id"] = group_id
            node["near_duplicate_kc_ids"] = [other for other in kc_ids if other != kc_id]
            node["dedup_status"] = status
            node["duplicate_match_type"] = match_type

    duplicate_kc_count = sum(len(group) for group in duplicate_groups)
    return {
        "semantic_merge_enabled": False,
        "duplicate_group_count": len(duplicate_groups),
        "exact_duplicate_group_count": exact_group_count,
        "near_duplicate_group_count": near_group_count,
        "duplicate_kc_count": duplicate_kc_count,
        "unique_kc_count": len(nodes) - duplicate_kc_count,
        "near_duplicate_jaccard_threshold": threshold,
        "near_duplicate_min_tokens": min_tokens,
    }


def _macro_centrality_scores(macro_spine: dict) -> dict[str, float]:
    macro_ids = [m["macro_id"] for m in macro_spine.get("macro_nodes", [])]
    neighbors = {macro_id: [] for macro_id in macro_ids}
    for edge in macro_spine.get("macro_edges", []):
        source = edge.get("source")
        target = edge.get("target")
        if source in neighbors and target in neighbors and source != target:
            neighbors[source].append(target)
            neighbors[target].append(source)
    max_degree = max((len(set(v)) for v in neighbors.values()), default=0)
    scores = {}
    for macro_id, linked in neighbors.items():
        degree = len(set(linked))
        degree_score = _log_score(degree, max_degree)
        entropy = _entropy_score(linked, len(macro_ids))
        scores[macro_id] = round(0.7 * degree_score + 0.3 * entropy, 4)
    return scores


def _graph_connectivity_scores(
    kc_nodes: list[dict],
    reasoning_edges: list[dict],
    macro_spine: dict,
) -> dict[str, float]:
    macro_count = len(macro_spine.get("macro_nodes", []))
    by_kc = {node["kc_id"]: node for node in kc_nodes}
    neighbors = {node["kc_id"]: [] for node in kc_nodes}
    for edge in reasoning_edges:
        source = edge.get("source")
        target = edge.get("target")
        if source in neighbors and target in neighbors and source != target:
            neighbors[source].append(target)
            neighbors[target].append(source)
    max_degree = max((len(set(v)) for v in neighbors.values()), default=0)
    scores = {}
    for kc_id, linked_ids in neighbors.items():
        degree = len(set(linked_ids))
        degree_score = _log_score(degree, max_degree)
        neighbor_macros = [
            by_kc[nid]["macro_id"]
            for nid in set(linked_ids)
            if nid in by_kc
        ]
        entropy = _entropy_score(neighbor_macros, macro_count)
        scores[kc_id] = round(0.7 * degree_score + 0.3 * entropy, 4)
    return scores


def _log_score(degree: int, max_degree: int) -> float:
    if degree <= 0 or max_degree <= 0:
        return 0.0
    return math.log(1 + degree) / math.log(1 + max_degree)


def _entropy_score(items: list[str], total_categories: int) -> float:
    if not items or total_categories <= 1:
        return 0.0
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    return entropy / math.log(total_categories)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("Embedding vector has zero norm.")
    return dot / (norm_a * norm_b)


def _require_score(scores: dict, key: str, kc_id: str) -> None:
    if key not in scores:
        raise ValueError(f"KC {kc_id} missing importance score: {key}")


def _score_1_5(raw: dict, key: str, kc_id: str) -> int:
    try:
        value = int(raw[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"KC {kc_id} missing integer {key}.") from exc
    if value < 1 or value > 5:
        raise ValueError(f"KC {kc_id} {key} must be in [1, 5], got {value}.")
    return value


def _short_label(claim: str) -> str:
    words = claim.strip().split()
    return " ".join(words[:10]) + ("..." if len(words) > 10 else "")


def _valid_type(value: object) -> str | None:
    return valid_kc_type(value, VALID_KC_TYPES)


def _valid_importance(value: object) -> str | None:
    text = str(value or "").strip()
    return text if text in {"critical", "normal"} else None


def _source_span_ids(candidate: dict) -> list[str]:
    out = []
    for key in ("unit_id", "source_chunk_id", "section_id", "candidate_id"):
        value = str(candidate.get(key, "")).strip()
        if value and value not in out:
            out.append(value)
    return out


def _candidate_evidence_items(candidate: dict, evidence_text: str, idx: int) -> list[dict]:
    raw_items = candidate.get("evidence_items")
    if raw_items is not None:
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError(f"KC candidate {candidate.get('candidate_id')} evidence_items must be a non-empty list when provided.")
        out = []
        for item_idx, item in enumerate(raw_items, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"KC candidate {candidate.get('candidate_id')} evidence_items[{item_idx}] must be an object.")
            text = str(item.get("text", "")).strip()
            if not text:
                raise ValueError(f"KC candidate {candidate.get('candidate_id')} evidence_items[{item_idx}] has empty text.")
            out.append(dict(item))
        return out
    return [
        {
            "section": candidate.get("section", ""),
            "span_id": candidate.get("section_id", "") or candidate.get("candidate_id", f"C{idx}"),
            "text": evidence_text,
        }
    ]


def _string_list(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    out = []
    for value in values:
        text = str(value).strip()
        if text:
            out.append(text)
    return out


def _normalize_claim_for_duplicate_check(text: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    tokens = re.findall(r"[a-z0-9]+", normalized)
    return " ".join(tokens)


def _token_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _kc_sort_key(kc_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", kc_id)
    return (int(match.group(1)) if match else 10**9, kc_id)


def _infer_type_from_macro(role: str) -> str:
    r = role.lower()
    if "problem" in r or "motivation" in r:
        return "problem"
    if "dataset" in r or "resource" in r:
        return "dataset"
    if "method" in r or "mechanism" in r or "module" in r:
        return "method"
    if "result" in r or "experiment" in r or "ablation" in r or "analysis" in r:
        return "result"
    if "limitation" in r:
        return "limitation"
    return "conclusion"


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None and raw.strip() else default
    except ValueError:
        value = default
    return max(minimum, value)


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    try:
        value = float(raw) if raw is not None and raw.strip() else default
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))
