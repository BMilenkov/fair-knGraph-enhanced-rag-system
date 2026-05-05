"""Fetch entity demographics from Wikidata SPARQL for fairness evaluation.

Uses Wikidata properties:
  P21: sex or gender
  P27: country of citizenship
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

GENDER_MAP = {
    "Q6581097": "male",
    "Q6581072": "female",
    "Q1052281": "transgender_female",
    "Q2449503": "transgender_male",
    "Q48270": "non_binary",
}

EUROPEAN_COUNTRIES = {
    "United Kingdom", "France", "Germany", "Italy", "Spain", "Netherlands",
    "Belgium", "Sweden", "Norway", "Denmark", "Finland", "Austria",
    "Switzerland", "Portugal", "Ireland", "Poland", "Czech Republic",
    "Greece", "Hungary", "Romania", "Bulgaria", "Croatia", "Slovakia",
    "Slovenia", "Estonia", "Latvia", "Lithuania", "Luxembourg", "Malta",
    "Cyprus", "Iceland", "Serbia", "Montenegro", "Albania",
    "North Macedonia", "Bosnia and Herzegovina", "Moldova", "Ukraine",
    "Belarus", "Russia",
}


@dataclass
class EntityDemographics:
    """Demographic attributes for a Wikidata entity."""
    qid: str
    label: str = ""
    gender: str | None = None
    country: str | None = None


def _run_sparql(query: str, max_retries: int = 3) -> list[dict]:
    """Execute a SPARQL query against Wikidata with retries."""
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "FairKGRAG/0.2 (academic research project)",
    }
    for attempt in range(max_retries):
        try:
            resp = requests.get(
                WIKIDATA_SPARQL_URL,
                params={"query": query},
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json().get("results", {}).get("bindings", [])
        except (requests.RequestException, ValueError) as e:
            logger.warning("SPARQL attempt %d failed: %s", attempt + 1, e)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return []


def classify_geo_group(country: str | None) -> str:
    """Classify country as 'european' or 'non_european' (RAG Fairness paper)."""
    if not country:
        return "unknown"
    return "european" if country in EUROPEAN_COUNTRIES else "non_european"


def fetch_demographics_batch(qids: list[str]) -> dict[str, EntityDemographics]:
    """Fetch gender and country from Wikidata for a list of QIDs.

    Batches of 50 to respect SPARQL limits.
    """
    if not qids:
        return {}

    results: dict[str, EntityDemographics] = {}

    for i in range(0, len(qids), 50):
        batch = qids[i:i + 50]
        values = " ".join(f"wd:{qid}" for qid in batch)

        query = f"""
        SELECT ?entity ?entityLabel ?gender ?countryLabel
        WHERE {{
            VALUES ?entity {{ {values} }}
            OPTIONAL {{ ?entity wdt:P21 ?gender . }}
            OPTIONAL {{ ?entity wdt:P27 ?country . }}
            SERVICE wikibase:label {{
                bd:serviceParam wikibase:language "en" .
            }}
        }}
        """

        for binding in _run_sparql(query):
            qid = binding["entity"]["value"].split("/")[-1]
            label = binding.get("entityLabel", {}).get("value", "")

            gender = None
            if "gender" in binding:
                gender_qid = binding["gender"]["value"].split("/")[-1]
                gender = GENDER_MAP.get(gender_qid, "other")

            country = None
            if "countryLabel" in binding:
                country = binding["countryLabel"]["value"]

            results[qid] = EntityDemographics(
                qid=qid, label=label, gender=gender, country=country,
            )

        if i + 50 < len(qids):
            time.sleep(1.0)

    return results


def demographics_to_dicts(demographics: dict[str, EntityDemographics]) -> list[dict]:
    """Serialize demographics to JSON-compatible dicts."""
    return [
        {
            "qid": d.qid,
            "label": d.label,
            "gender": d.gender,
            "country": d.country,
            "geo_group": classify_geo_group(d.country),
        }
        for d in demographics.values()
    ]
