"""Fetch entity demographics from Wikidata SPARQL endpoint.

Uses Wikidata properties:
- P21: sex or gender
- P27: country of citizenship
- P19: place of birth
- P106: occupation

This provides ground-truth demographic attributes for fairness evaluation,
avoiding noisy LLM-based annotation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# Mapping of gender QIDs to readable labels
GENDER_MAP = {
    "Q6581097": "male",
    "Q6581072": "female",
    "Q1052281": "transgender_female",
    "Q2449503": "transgender_male",
    "Q48270": "non_binary",
}

# Continent mapping for geographic fairness
CONTINENT_MAP = {
    "Q46": "Africa",
    "Q48": "Asia",
    "Q49": "North_America",
    "Q18": "South_America",
    "Q46": "Africa",
    "Q15": "Africa",
    "Q538": "Oceania",
    "Q13": "Europe",
}


@dataclass
class EntityDemographics:
    """Demographic attributes for a Wikidata entity.

    Attributes:
        qid: Wikidata QID (e.g., "Q42").
        label: Entity name/label.
        gender: Gender label (male, female, etc.) or None.
        country: Country of citizenship or None.
        birthplace: Place of birth or None.
        is_person: Whether the entity represents a person.
    """

    qid: str
    label: str = ""
    gender: str | None = None
    country: str | None = None
    birthplace: str | None = None
    is_person: bool = False


def _run_sparql_query(query: str, max_retries: int = 3) -> list[dict[str, Any]]:
    """Execute a SPARQL query against Wikidata.

    Args:
        query: SPARQL query string.
        max_retries: Number of retry attempts on failure.

    Returns:
        List of result bindings.
    """
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "FairKGRAG/0.1 (academic research project)",
    }

    for attempt in range(max_retries):
        try:
            response = requests.get(
                WIKIDATA_SPARQL_URL,
                params={"query": query},
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", {}).get("bindings", [])
        except (requests.RequestException, ValueError) as e:
            logger.warning(f"SPARQL query attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"SPARQL query failed after {max_retries} attempts")
                return []


def fetch_demographics_batch(qids: list[str]) -> dict[str, EntityDemographics]:
    """Fetch demographic attributes for a batch of Wikidata entities.

    Args:
        qids: List of Wikidata QIDs (e.g., ["Q42", "Q937"]).

    Returns:
        Dictionary mapping QID to EntityDemographics.
    """
    if not qids:
        return {}

    # Wikidata SPARQL has limits; process in batches of 50
    results: dict[str, EntityDemographics] = {}
    batch_size = 50

    for i in range(0, len(qids), batch_size):
        batch = qids[i : i + batch_size]
        values = " ".join(f"wd:{qid}" for qid in batch)

        query = f"""
        SELECT ?entity ?entityLabel ?genderLabel ?countryLabel ?birthplaceLabel
               ?gender ?country ?birthplace
        WHERE {{
            VALUES ?entity {{ {values} }}
            OPTIONAL {{ ?entity wdt:P21 ?gender . }}
            OPTIONAL {{ ?entity wdt:P27 ?country . }}
            OPTIONAL {{ ?entity wdt:P19 ?birthplace . }}
            SERVICE wikibase:label {{
                bd:serviceParam wikibase:language "en" .
            }}
        }}
        """

        bindings = _run_sparql_query(query)

        for binding in bindings:
            qid = binding["entity"]["value"].split("/")[-1]
            label = binding.get("entityLabel", {}).get("value", "")

            # Parse gender
            gender = None
            if "gender" in binding:
                gender_qid = binding["gender"]["value"].split("/")[-1]
                gender = GENDER_MAP.get(gender_qid, "other")

            # Parse country
            country = None
            if "countryLabel" in binding:
                country = binding["countryLabel"]["value"]

            # Parse birthplace
            birthplace = None
            if "birthplaceLabel" in binding:
                birthplace = binding["birthplaceLabel"]["value"]

            results[qid] = EntityDemographics(
                qid=qid,
                label=label,
                gender=gender,
                country=country,
                birthplace=birthplace,
                is_person=gender is not None,
            )

        # Respect Wikidata rate limits
        if i + batch_size < len(qids):
            time.sleep(1.0)

    return results


def classify_geographic_group(country: str | None) -> str:
    """Classify a country into a geographic group for fairness evaluation.

    Following the RAG Fairness paper, groups entities into
    European vs. non-European for geographic fairness analysis.

    Args:
        country: Country name string.

    Returns:
        Geographic group label: "european", "non_european", or "unknown".
    """
    if not country:
        return "unknown"

    european_countries = {
        "United Kingdom", "France", "Germany", "Italy", "Spain", "Netherlands",
        "Belgium", "Sweden", "Norway", "Denmark", "Finland", "Austria",
        "Switzerland", "Portugal", "Ireland", "Poland", "Czech Republic",
        "Greece", "Hungary", "Romania", "Bulgaria", "Croatia", "Slovakia",
        "Slovenia", "Estonia", "Latvia", "Lithuania", "Luxembourg", "Malta",
        "Cyprus", "Iceland", "Serbia", "Montenegro", "Albania",
        "North Macedonia", "Bosnia and Herzegovina", "Moldova", "Ukraine",
        "Belarus", "Russia",
    }

    if country in european_countries:
        return "european"
    return "non_european"


def demographics_to_dicts(
    demographics: dict[str, EntityDemographics],
) -> list[dict]:
    """Convert demographics to serializable dictionaries.

    Args:
        demographics: Mapping of QID to EntityDemographics.

    Returns:
        List of dictionaries.
    """
    return [
        {
            "qid": d.qid,
            "label": d.label,
            "gender": d.gender,
            "country": d.country,
            "birthplace": d.birthplace,
            "is_person": d.is_person,
            "geo_group": classify_geographic_group(d.country),
        }
        for d in demographics.values()
    ]
