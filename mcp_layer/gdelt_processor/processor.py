"""GDELT processor: fetches/caches adverse media news and returns Evidence.

Relevance filtering:
  - Articles where the entity name appears in the title → conf=0.70 (relevant=True)
  - Articles where a risk keyword appears in the title → conf=0.55 (relevant=True)
  - All other articles → conf=0.30 (relevant=False, kept but down-weighted)

This addresses the ~76% noise rate in raw GDELT results.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from osint_swarm.data_sources import gdelt
from osint_swarm.entities import Evidence
from osint_swarm.utils.io import read_json, write_json

from app.investigation_errors import DataSourceError
from mcp_layer.base import DataSourceProcessor

if TYPE_CHECKING:
    from osint_swarm.entities import Entity

RISK_TITLE_KEYWORDS = re.compile(
    r"\b(fraud|investigat|penalt|fine[ds]?\b|violation|lawsuit|scandal|"
    r"misconduct|briber|corrupt|sanction|launder|settlement|"
    r"indictment|sec\b|enforc|guilty|charge[ds]?\b|convict|probe[ds]?\b|"
    r"subpoena|whistleblow|recall|class.action|regulat|sued\b|suing\b)",
    re.IGNORECASE,
)


def _score_relevance(title: str, entity_name: str) -> Tuple[float, bool]:
    """
    Score an article's relevance based on its title.

    Returns (confidence, is_relevant):
      - Entity name in title → (0.70, True)
      - Risk keyword in title → (0.55, True)
      - Neither → (0.30, False)
    """
    title_lower = title.lower()
    # Check entity name (use first word of name for matching, e.g. "Tesla" from "Tesla, Inc.")
    name_parts = [p.strip().lower() for p in re.split(r"[,.]", entity_name) if p.strip()]
    name_tokens = []
    for part in name_parts:
        for word in part.split():
            if len(word) >= 3:
                name_tokens.append(word)

    entity_in_title = any(tok in title_lower for tok in name_tokens)
    risk_in_title = bool(RISK_TITLE_KEYWORDS.search(title))

    if entity_in_title and risk_in_title:
        return 0.75, True
    if entity_in_title:
        return 0.70, True
    if risk_in_title:
        return 0.55, True
    return 0.30, False


def _articles_to_evidence(
    articles: List[Dict[str, Any]],
    entity_id: str,
    entity_name: str,
    raw_location: Optional[str] = None,
) -> List[Evidence]:
    """Convert GDELT article records to Evidence list with relevance scoring."""
    out: List[Evidence] = []
    for i, article in enumerate(articles):
        if not isinstance(article, dict):
            continue

        title = (article.get("title") or "").strip()
        url = (article.get("url") or "").strip()
        seen_date = (article.get("seendate") or "").strip()
        domain = (article.get("domain") or "").strip()
        language = (article.get("language") or "").strip()
        source_country = (article.get("sourcecountry") or "").strip()

        if not url or not title:
            continue

        date_str = ""
        if seen_date:
            raw = seen_date.replace("T", "").replace("Z", "").replace("-", "")
            if len(raw) >= 8:
                date_str = f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"

        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        ev_id = f"{entity_id.split('_')[0]}_gdelt_{url_hash}"

        confidence, is_relevant = _score_relevance(title, entity_name)

        summary = title if title else f"News article about {entity_name}"

        out.append(
            Evidence(
                evidence_id=ev_id,
                entity_id=entity_id,
                date=date_str,
                source_type="news_article",
                risk_category="network",
                summary=summary[:5000],
                source_uri=url,
                raw_location=raw_location,
                confidence=confidence,
                attributes={
                    "domain": domain,
                    "language": language,
                    "source_country": source_country,
                    "gdelt_rank": i + 1,
                    "relevant": is_relevant,
                },
            )
        )
    return out


class GdeltProcessor(DataSourceProcessor):
    """MCP processor for GDELT adverse media; uses osint_swarm.data_sources.gdelt."""

    def __init__(self, data_root: Optional[Path] = None):
        self.data_root = Path(data_root) if data_root else Path("data")
        self._raw_dir = self.data_root / "raw" / "gdelt"

    @property
    def source_id(self) -> str:
        return "gdelt"

    def _slug_for_entity(self, entity: "Entity") -> str:
        """Filesystem-safe slug from entity name."""
        return entity.name.lower().split(",")[0].strip().replace(" ", "_").replace(".", "")

    def get_evidence_for_entity(self, entity: "Entity") -> List[Evidence]:
        entity_id = entity.entity_id
        slug = self._slug_for_entity(entity)

        cache_path = self._raw_dir / f"news_{slug}.json"
        if cache_path.exists():
            payload = read_json(cache_path)
            raw_location = str(cache_path)
        else:
            try:
                payload = gdelt.fetch_news_for_entity(entity.name)
                self._raw_dir.mkdir(parents=True, exist_ok=True)
                write_json(cache_path, payload)
                raw_location = str(cache_path)
            except gdelt.GdeltError as exc:
                raise DataSourceError(
                    "GDELT data unavailable (cache missing and live fetch failed). "
                    "Retry later or pre-cache GDELT results under data/raw/gdelt/."
                ) from exc

        articles = gdelt.extract_article_records(payload)
        # Keep only English-language articles (filters cached non-English entries too)
        articles = [a for a in articles if (a.get("language") or "").lower() in ("english", "")]
        return _articles_to_evidence(articles, entity_id, entity.name, raw_location=raw_location)
