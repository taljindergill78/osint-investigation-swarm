"""
OFAC SDN (Specially Designated Nationals) connector.

Downloads and parses the US Treasury Office of Foreign Assets Control
Specially Designated Nationals (SDN) list.

Source: https://www.treasury.gov/ofac/downloads/sdn.xml
  - Free, no authentication, no API key required
  - Updated regularly by the US Treasury
  - ~2-3 MB XML file; cached locally after first download

The SDN list contains individuals, entities, vessels, and aircraft that
US persons are prohibited from transacting with. Matching an entity name
against this list is a core component of AML/KYC due diligence.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

SDN_URL = "https://www.treasury.gov/ofac/downloads/sdn.xml"
SDN_NAMESPACE = "https://sanctionssearch.ofac.treas.gov/"


class OfacError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def download_sdn_xml(cache_path: Path, user_agent: str = "OSINT-Swarm research@asu.edu") -> Path:
    """Download the OFAC SDN XML to cache_path. Returns the path."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(SDN_URL, headers={"User-Agent": user_agent}, timeout=60, stream=True)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise OfacError(f"Failed to download OFAC SDN XML: {exc}") from exc

    with open(cache_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)
    return cache_path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _strip_ns(tree: ET.Element) -> None:
    """Strip XML namespace prefixes from all tags in-place."""
    for elem in tree.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]


def _normalize(s: str) -> str:
    """Lowercase, strip punctuation, collapse spaces, remove common legal suffixes."""
    s = s.lower()
    s = re.sub(r"[,.\-&'/()]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for suffix in (" inc", " llc", " ltd", " corp", " co", " company", " the"):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    return s


def parse_sdn_entries(xml_path: Path) -> List[Dict[str, Any]]:
    """
    Parse an OFAC SDN XML file and return a list of SDN entry dicts.

    Each dict has keys:
      uid, name, sdn_type, programs (list), aka_names (list), remarks,
      _norms (list[str])  — pre-normalized name variants for fast matching
    """
    try:
        tree = ET.parse(str(xml_path))
    except ET.ParseError as exc:
        raise OfacError(f"Failed to parse OFAC SDN XML at {xml_path}: {exc}") from exc

    root = tree.getroot()
    _strip_ns(root)

    entries: List[Dict[str, Any]] = []
    for entry in root.findall("sdnEntry"):
        uid = entry.findtext("uid") or ""
        first = (entry.findtext("firstName") or "").strip()
        last = (entry.findtext("lastName") or "").strip()
        name = f"{first} {last}".strip() if first else last

        sdn_type = (entry.findtext("sdnType") or "").strip()
        remarks = (entry.findtext("remarks") or "").strip()

        programs: List[str] = []
        prog_list = entry.find("programList")
        if prog_list is not None:
            for prog in prog_list.findall("program"):
                if prog.text:
                    programs.append(prog.text.strip())

        aka_names: List[str] = []
        aka_list = entry.find("akaList")
        if aka_list is not None:
            for aka in aka_list.findall("aka"):
                aka_first = (aka.findtext("firstName") or "").strip()
                aka_last = (aka.findtext("lastName") or "").strip()
                aka_name = f"{aka_first} {aka_last}".strip() if aka_first else aka_last
                if aka_name:
                    aka_names.append(aka_name)

        if name:
            all_names = [name] + aka_names
            norms = [_normalize(n) for n in all_names if n]
            entries.append({
                "uid": uid,
                "name": name,
                "sdn_type": sdn_type,
                "programs": programs,
                "aka_names": aka_names,
                "remarks": remarks,
                "_norms": norms,
            })
    return entries


# ---------------------------------------------------------------------------
# Matching — no regex, uses word-set containment only
# ---------------------------------------------------------------------------

def _word_set(s: str) -> frozenset:
    return frozenset(s.split())


def _names_match(qterm: str, sdn_norm: str) -> bool:
    """
    True if qterm and sdn_norm refer to the same entity.

    Rules:
    1. Exact match after normalization.
    2. All words of qterm (≥5 chars) appear in sdn_norm's word set.
    3. All words of sdn_norm (≥5 chars) appear in qterm's word set.

    Rule 3 handles shorter trading names inside longer legal names
    (e.g. querying "The Boeing Company" matches SDN entry "BOEING").
    No regex is used — safe on all Python versions.
    """
    if qterm == sdn_norm:
        return True
    qwords = _word_set(qterm)
    swords = _word_set(sdn_norm)
    if len(qterm) >= 5 and qwords and qwords.issubset(swords):
        return True
    if len(sdn_norm) >= 5 and swords and swords.issubset(qwords):
        return True
    return False


def _terms_match(qterm: str, sdn_norm: str) -> bool:
    """Backward-compatible alias retained for existing tests/callers."""
    return _names_match(qterm, sdn_norm)


def search_entries(
    entries: List[Dict[str, Any]],
    entity_name: str,
    aliases: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Search parsed SDN entries for the entity name and aliases.

    Returns a list of matching SDN entry dicts (may be empty for a clean result).
    False positives are possible due to name similarity; callers should treat
    matches as flags for human review, not automatic disqualification.
    """
    query_terms: List[str] = [_normalize(entity_name)]
    for alias in (aliases or []):
        if len(alias.strip()) < 3:
            continue
        n = _normalize(alias)
        if n and n not in query_terms:
            query_terms.append(n)

    hits: List[Dict[str, Any]] = []
    seen_uids: set = set()

    for entry in entries:
        if entry["uid"] in seen_uids:
            continue

        sdn_norms: List[str] = entry.get("_norms") or [
            _normalize(n) for n in ([entry["name"]] + entry["aka_names"]) if n
        ]

        for qterm in query_terms:
            for sdn_norm in sdn_norms:
                if sdn_norm and _names_match(qterm, sdn_norm):
                    seen_uids.add(entry["uid"])
                    hits.append(entry)
                    break
            else:
                continue
            break

    return hits
