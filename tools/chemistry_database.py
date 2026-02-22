"""
title: Chemical Compound Intelligence System (OpenRouter)
author: Marios Adamidis (FORTHought Lab)
version: 1.0.0
description: |
    Comprehensive search for chemical compounds in PubChem and CAS Common Chemistry.
    Uses OpenRouter (OpenAI-compatible) for name translation to English.
    Provides structural data, physical properties, synonyms, and visualization.
    v3.2: moved internal methods out of Tools class to prevent OWUI tool leakage.
requirements: aiohttp, requests, pubchempy, pydantic
"""

import aiohttp
import asyncio
import json
import time
import random
import ssl
import urllib.parse
import logging
import re
import os
from typing import Dict, Any, List, Optional, Union

from pydantic import BaseModel, Field, ConfigDict, validator
import pubchempy as pcp
import requests

# Logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ============================================================================
# MODULE-LEVEL HELPERS  (invisible to OWUI tool scanner)
# ============================================================================


def clean_html_tags(text: Union[str, List[str]]) -> Union[str, List[str]]:
    """Removes HTML tags from text"""
    if isinstance(text, list):
        return [re.sub(r"<[^>]+>", "", str(item)) for item in text]
    return re.sub(r"<[^>]+>", "", str(text))


def extract_cas_from_text(text: str) -> List[str]:
    """Extracts CAS numbers from text using regular expressions"""
    if not text:
        return []
    pattern = r"\b\d{2,7}-\d{2}-\d{1}\b"
    return re.findall(pattern, str(text))


def _log(debug_logs: List[str], valves, message: str) -> None:
    """Internal function to collect logs"""
    timestamp = time.strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    debug_logs.append(log_entry)
    if valves.debug_mode:
        print(f"***ChemicalCompound_Search [{timestamp}] {message}")


def _get_openrouter_chat_url(valves) -> str:
    base = valves.openrouter_base_url.rstrip("/")
    return f"{base}/chat/completions"


def _get_openrouter_headers(valves) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {valves.openrouter_api_key.strip()}",
        "Content-Type": "application/json",
    }
    if valves.openrouter_site_url.strip():
        headers["HTTP-Referer"] = valves.openrouter_site_url.strip()
    if valves.openrouter_app_name.strip():
        headers["X-Title"] = valves.openrouter_app_name.strip()
    return headers


async def _openrouter_chat(
    valves, debug_logs: List[str], messages: List[Dict[str, str]]
) -> Dict[str, Any]:
    """OpenRouter Chat Completions (OpenAI-compatible) with retries."""
    if not valves.openrouter_api_key.strip():
        raise ValueError(
            "OPENROUTER_API_KEY is missing (set env var or valves.openrouter_api_key)."
        )

    url = _get_openrouter_chat_url(valves)
    headers = _get_openrouter_headers(valves)

    payload = {
        "model": valves.openrouter_model.strip(),
        "messages": messages,
        "temperature": 0.1,
        "top_p": 0.9,
        "stream": False,
    }

    async with aiohttp.ClientSession() as session:
        for attempt in range(valves.max_retries):
            try:
                async with session.post(
                    url, headers=headers, json=payload, timeout=30
                ) as resp:
                    text = await resp.text()
                    if resp.status == 200:
                        return json.loads(text)

                    if resp.status in (429, 502, 503, 504):
                        if attempt < valves.max_retries - 1:
                            wait_time = (
                                valves.delay * (attempt + 1) * random.uniform(0.5, 1.5)
                            )
                            _log(
                                debug_logs,
                                valves,
                                f"OpenRouter status {resp.status}, attempt {attempt + 1}/{valves.max_retries}, "
                                f"waiting {wait_time:.1f}s",
                            )
                            await asyncio.sleep(wait_time)
                            continue

                    raise RuntimeError(f"OpenRouter error {resp.status}: {text}")

            except (aiohttp.ClientError, asyncio.TimeoutError, ssl.SSLError) as e:
                if attempt < valves.max_retries - 1:
                    wait_time = valves.delay * (attempt + 1) * random.uniform(0.5, 1.5)
                    _log(
                        debug_logs,
                        valves,
                        f"OpenRouter error (attempt {attempt + 1}/{valves.max_retries}): {str(e)}, "
                        f"waiting {wait_time:.1f}s",
                    )
                    await asyncio.sleep(wait_time)
                else:
                    raise

    raise RuntimeError("OpenRouter request failed after all retries")


async def _translate_to_english(
    valves, debug_logs: List[str], text: str
) -> Dict[str, str]:
    """
    Translates text to English using OpenRouter.
    Returns dict: {"translated": str, "model_used": str}
    """
    text = (text or "").strip()
    if not text:
        return {
            "translated": text,
            "model_used": valves.openrouter_model.strip(),
        }

    # Heuristic: if already English-ish, skip translation
    if re.match(r'^[a-zA-Z0-9\s\-\(\),.\[\]{}\'"]+$', text) and any(
        c.isalpha() for c in text
    ):
        _log(debug_logs, valves, "Text is already in English, translation not required")
        return {
            "translated": text,
            "model_used": valves.openrouter_model.strip(),
        }

    _log(
        debug_logs,
        valves,
        f"Translating via OpenRouter model: {valves.openrouter_model}",
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You translate chemical compound names to English. "
                "Return ONLY the English name. No explanations. No extra text."
            ),
        },
        {
            "role": "user",
            "content": f"Translate this chemical compound name to English: {text}",
        },
    ]

    try:
        r = await _openrouter_chat(valves, debug_logs, messages)
        content = (
            r.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        )

        # OpenRouter responses often include the selected model at top-level "model"
        model_used = str(r.get("model") or valves.openrouter_model).strip()

        # Cleanup
        content = re.sub(r"^[^a-zA-Z0-9]+", "", content)
        content = re.sub(r"[^a-zA-Z0-9\s\-\(\),.<>]+$", "", content)

        if content:
            _log(debug_logs, valves, f"Translation: '{text}' -> '{content}'")
            return {"translated": content, "model_used": model_used}

        _log(
            debug_logs,
            valves,
            "OpenRouter returned empty translation, falling back to original",
        )
        return {"translated": text, "model_used": model_used}

    except Exception as e:
        _log(debug_logs, valves, f"Translation error: {str(e)}")
        return {
            "translated": text,
            "model_used": valves.openrouter_model.strip(),
        }


async def _fetch_cas_common_chemistry_async(
    valves, debug_logs: List[str], cas_rn: str
) -> Dict[str, Any]:
    """Asynchronous request to CAS Common Chemistry API"""
    if not valves.use_cas or not valves.cas_api_key:
        _log(debug_logs, valves, "CAS search disabled or API key missing")
        return {}

    data = {
        "Formula": "",
        "Name": "",
        "SMILES": "",
        "Melting": "",
        "Boiling": "",
        "Density": "",
        "Synonyms": "",
        "Image": "",
    }

    cas_rn_clean = cas_rn.strip().replace(" ", "")
    if not cas_rn_clean:
        return data

    url = f"https://commonchemistry.cas.org/api/detail?cas_rn={urllib.parse.quote(cas_rn_clean)}"
    headers = {"X-API-KEY": valves.cas_api_key}

    async with aiohttp.ClientSession() as session:
        for attempt in range(valves.max_retries):
            try:
                async with session.get(
                    url, headers=headers, timeout=valves.timeout
                ) as response:
                    if response.status == 200:
                        j = await response.json()
                        data["Formula"] = clean_html_tags(j.get("molecularFormula", ""))
                        data["Name"] = clean_html_tags(j.get("name", ""))
                        data["SMILES"] = clean_html_tags(j.get("canonicalSmile", ""))
                        data["Image"] = j.get("image", "")

                        props = j.get("experimentalProperties", [])
                        for p in props:
                            name = str(p.get("name", "")).lower()
                            val = p.get("property", "")
                            if "melting point" in name or "melting" in name:
                                data["Melting"] = val
                            if "boiling point" in name or "boiling" in name:
                                data["Boiling"] = val
                            if "density" in name:
                                data["Density"] = val

                        syns = clean_html_tags(j.get("synonyms", []))
                        if isinstance(syns, list):
                            data["Synonyms"] = " | ".join(syns[:50])
                        else:
                            data["Synonyms"] = str(syns)

                        _log(
                            debug_logs,
                            valves,
                            f"Successfully retrieved CAS data for {cas_rn}: {data['Name']}",
                        )
                        return data

                    if (
                        response.status in (429, 502, 503, 504)
                        and attempt < valves.max_retries - 1
                    ):
                        wait_time = (
                            valves.delay * (attempt + 1) * random.uniform(0.5, 1.5)
                        )
                        _log(
                            debug_logs,
                            valves,
                            f"CAS API status {response.status} for {cas_rn}, "
                            f"attempt {attempt + 1}/{valves.max_retries}, waiting {wait_time:.1f}s",
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    _log(
                        debug_logs,
                        valves,
                        f"CAS API unexpected status {response.status} for {cas_rn}",
                    )
                    return data

            except (aiohttp.ClientError, asyncio.TimeoutError, ssl.SSLError) as e:
                if attempt < valves.max_retries - 1:
                    wait_time = valves.delay * (attempt + 1) * random.uniform(0.5, 1.5)
                    _log(
                        debug_logs,
                        valves,
                        f"CAS API error (attempt {attempt + 1}/{valves.max_retries}) for {cas_rn}: {str(e)}, "
                        f"waiting {wait_time:.1f}s",
                    )
                    await asyncio.sleep(wait_time)
                else:
                    _log(
                        debug_logs,
                        valves,
                        f"CAS API all attempts failed for {cas_rn}: {str(e)}",
                    )
                    return data
            except Exception as e:
                _log(debug_logs, valves, f"Unexpected CAS error for {cas_rn}: {str(e)}")
                return data

    return data


def _fetch_pubchem_data_sync(
    valves, debug_logs: List[str], identifier: str
) -> Dict[str, Any]:
    """Synchronous request to PubChem"""
    if not valves.use_pubchem:
        _log(debug_logs, valves, "PubChem search disabled")
        return {}

    identifier_lower = (
        identifier.lower().strip() if isinstance(identifier, str) else str(identifier)
    )
    if not identifier_lower:
        return {}

    data: Dict[str, Any] = {
        "Formula": "",
        "IUPAC": "",
        "Synonyms": "",
        "CAS_Found": "",
        "Melting": "",
        "Boiling": "",
        "Density": "",
        "Color_Form": "",
        "MolecularWeight": "",
        "Solubility": "",
        "LogP": "",
        "PolarSurfaceArea": "",
        "SMILES": "",
        "InChI": "",
        "CID": "",
    }

    for attempt in range(valves.max_retries):
        try:
            compounds = pcp.get_compounds(
                identifier_lower, namespace="name", timeout=valves.timeout
            )

            if not compounds:
                _log(
                    debug_logs,
                    valves,
                    f"PubChem: no compounds found for {identifier_lower}",
                )
                return data

            c = compounds[0]
            data["CID"] = str(c.cid) if getattr(c, "cid", None) else ""
            data["Formula"] = c.molecular_formula or ""
            data["IUPAC"] = c.iupac_name or ""
            data["MolecularWeight"] = (
                str(c.molecular_weight) if c.molecular_weight else ""
            )
            data["SMILES"] = c.canonical_smiles or ""
            data["InChI"] = c.inchi or ""

            try:
                if hasattr(c, "xlogp") and c.xlogp is not None:
                    data["LogP"] = str(c.xlogp)
                if hasattr(c, "tpsa") and c.tpsa is not None:
                    data["PolarSurfaceArea"] = str(c.tpsa)
            except Exception:
                pass

            try:
                synonyms = c.synonyms or []
                data["Synonyms"] = " | ".join(synonyms[:50])

                all_cas = set()
                for s in synonyms:
                    for found in extract_cas_from_text(str(s)):
                        all_cas.add(found)
                if all_cas:
                    data["CAS_Found"] = " | ".join(sorted(all_cas))
            except Exception as e:
                _log(debug_logs, valves, f"Error retrieving synonyms: {e}")

            try:
                props = pcp.get_properties(
                    ["MeltingPoint", "BoilingPoint", "Density", "Solubility"],
                    c.cid,
                    "cid",
                )
                if props:
                    prop_data = props[0]
                    data["Melting"] = str(prop_data.get("MeltingPoint", "") or "")
                    data["Boiling"] = str(prop_data.get("BoilingPoint", "") or "")
                    data["Density"] = str(prop_data.get("Density", "") or "")
                    data["Solubility"] = str(prop_data.get("Solubility", "") or "")
            except Exception:
                pass

            _log(
                debug_logs,
                valves,
                f"Successfully retrieved PubChem data for {identifier_lower}: {data['IUPAC']}",
            )
            return data

        except (pcp.PubChemHTTPError, requests.exceptions.RequestException) as e:
            if "PUGREST.NotFound" in str(e):
                _log(
                    debug_logs,
                    valves,
                    f"PubChem: compound not found for {identifier_lower}",
                )
                return data
            if attempt < valves.max_retries - 1:
                _log(
                    debug_logs,
                    valves,
                    f"PubChem error (attempt {attempt + 1}/{valves.max_retries}): {str(e)}",
                )
                time.sleep(valves.delay * (attempt + 1) * random.uniform(0.5, 1.5))
            else:
                _log(
                    debug_logs,
                    valves,
                    f"PubChem: all attempts failed for {identifier_lower}: {str(e)}",
                )
                return data
        except Exception as e:
            if attempt < valves.max_retries - 1:
                _log(
                    debug_logs,
                    valves,
                    f"Unexpected PubChem error (attempt {attempt + 1}/{valves.max_retries}): {str(e)}",
                )
                time.sleep(valves.delay * (attempt + 1) * random.uniform(0.5, 1.5))
            else:
                _log(
                    debug_logs,
                    valves,
                    f"PubChem: all attempts failed for {identifier_lower}: {str(e)}",
                )
                return data

    return data


async def _get_compound_data(
    valves, debug_logs: List[str], compound_name: str
) -> Dict[str, Any]:
    """Main function to retrieve compound data"""
    result: Dict[str, Any] = {
        "query": compound_name,
        "pubchem_data": {},
        "cas_data": {},
        "combined_data": {},
        "success": False,
        "llm_provider": "openrouter",
        "used_model": valves.openrouter_model.strip(),
        "llm_url": _get_openrouter_chat_url(valves),
        "translated_name": compound_name,
    }

    # Translate name to English
    t = await _translate_to_english(valves, debug_logs, compound_name)
    translated_name = t["translated"]
    result["translated_name"] = translated_name
    result["used_model"] = t.get("model_used", result["used_model"])
    _log(
        debug_logs,
        valves,
        f"Original name: '{compound_name}', Translated name: '{translated_name}'",
    )

    # PubChem
    if valves.use_pubchem:
        pubchem_data = await asyncio.to_thread(
            _fetch_pubchem_data_sync, valves, debug_logs, translated_name
        )
        result["pubchem_data"] = pubchem_data

    # CAS (optional)
    if valves.use_cas:
        cas_numbers: List[str] = []

        if result["pubchem_data"] and result["pubchem_data"].get("CAS_Found"):
            cas_numbers = [
                x.strip()
                for x in result["pubchem_data"]["CAS_Found"].split(" | ")
                if x.strip()
            ]
        elif translated_name and re.search(r"\b\d{2,7}-\d{2}-\d{1}\b", translated_name):
            m = re.search(r"\b\d{2,7}-\d{2}-\d{1}\b", translated_name)
            if m:
                cas_numbers = [m.group(0)]

        if cas_numbers:
            for cas_rn in cas_numbers[:3]:
                cas_data = await _fetch_cas_common_chemistry_async(
                    valves, debug_logs, cas_rn
                )
                result["cas_data"][cas_rn] = cas_data
        else:
            _log(
                debug_logs,
                valves,
                "CAS number not found in PubChem/name, skipping CAS search",
            )

    # Combine
    combined: Dict[str, Any] = {}
    if result["pubchem_data"]:
        combined.update(result["pubchem_data"])

    for cas_rn, cas_data in result["cas_data"].items():
        if cas_data and isinstance(cas_data, dict):
            for k, v in cas_data.items():
                if v and (k not in combined or not combined.get(k)):
                    combined[k] = v

    result["combined_data"] = combined
    result["success"] = bool(combined)
    return result


# ============================================================================
# TOOLS CLASS  — only the public tool remains here
# ============================================================================


class Tools:
    def __init__(self):
        self.valves = self.Valves()
        self.debug_logs: List[str] = []

    class Valves(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        # OpenRouter settings (OpenAI-compatible)
        openrouter_base_url: str = Field(
            "https://openrouter.ai/api/v1",
            description="OpenRouter base URL (OpenAI-compatible), e.g. https://openrouter.ai/api/v1",
        )
        openrouter_api_key: str = Field(
            default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""),
            description="OpenRouter API key (recommended to set via env OPENROUTER_API_KEY).",
        )
        openrouter_model: str = Field(
            "openrouter/auto",
            description="OpenRouter model slug, e.g. openrouter/auto or openai/gpt-4o-mini",
        )
        openrouter_site_url: str = Field(
            default_factory=lambda: os.getenv("OPENROUTER_SITE_URL", ""),
            description="Optional attribution header HTTP-Referer",
        )
        openrouter_app_name: str = Field(
            default_factory=lambda: os.getenv("OPENROUTER_APP_NAME", "FORTHought"),
            description="Optional attribution header X-Title",
        )

        # Data source settings
        use_pubchem: bool = Field(
            True, description="Use PubChem search to retrieve compound data."
        )
        use_cas: bool = Field(
            False, description="Use CAS Common Chemistry API to retrieve data."
        )
        cas_api_key: str = Field(
            "",
            description="API key for CAS Common Chemistry API (required if CAS search is enabled).",
        )

        # Runtime settings
        max_retries: int = Field(
            3, description="Maximum number of attempts for request errors."
        )
        delay: float = Field(
            0.1, description="Base delay between retries (in seconds)."
        )
        timeout: int = Field(15, description="Request timeout in seconds.")
        debug_mode: bool = Field(False, description="Show detailed logs for debugging.")

        @validator("cas_api_key")
        def validate_cas_api_key(cls, v, values):
            if values.get("use_cas") and not str(v).strip():
                raise ValueError("CAS API key required when CAS search is enabled")
            return v

    # MAIN FUNCTION FOR OPEN WEBUI — the ONE public tool
    async def get_chemical_compound_info(self, compound_name: str) -> str:
        """
        Retrieves detailed information about a chemical compound from PubChem and CAS.

        :param compound_name: Name of the chemical compound in any language
        :return: Structured report with compound data
        """
        self.debug_logs = []
        _log(
            self.debug_logs,
            self.valves,
            f"--- STARTING CHEMICAL COMPOUND SEARCH: {compound_name} ---",
        )

        try:
            result = await _get_compound_data(
                self.valves, self.debug_logs, compound_name
            )
            translated_name = result.get("translated_name", compound_name)

            output = f"# Chemical Information: {compound_name}\n\n"

            if result["success"]:
                data = result["combined_data"]

                # Basic Information
                output += "## Basic Characteristics\n"
                if data.get("Name") or data.get("IUPAC"):
                    output += f"**Name:** {data.get('Name', data.get('IUPAC', 'Not specified'))}\n"
                if data.get("Formula"):
                    output += f"**Molecular Formula:** {data.get('Formula')}\n"
                if data.get("MolecularWeight"):
                    output += (
                        f"**Molecular Weight:** {data.get('MolecularWeight')} g/mol\n"
                    )

                # Identifiers
                output += "\n## Identifiers\n"
                if data.get("SMILES"):
                    output += f"**SMILES:** `{data.get('SMILES')}`\n"
                if data.get("InChI"):
                    output += f"**InChI:** `{data.get('InChI')}`\n"
                if result["pubchem_data"].get("CID"):
                    cid = result["pubchem_data"].get("CID")
                    output += f"[**PubChem CID {cid}**](https://pubchem.ncbi.nlm.nih.gov/compound/{cid})\n"

                # Physical Properties
                has_properties = any(
                    data.get(prop)
                    for prop in [
                        "Melting",
                        "Boiling",
                        "Density",
                        "Solubility",
                        "LogP",
                        "PolarSurfaceArea",
                    ]
                )
                if has_properties:
                    output += "\n## Physical and Chemical Properties\n"
                    if data.get("Melting"):
                        output += f"**Melting Point:** {data.get('Melting')}\n"
                    if data.get("Boiling"):
                        output += f"**Boiling Point:** {data.get('Boiling')}\n"
                    if data.get("Density"):
                        output += f"**Density:** {data.get('Density')}\n"
                    if data.get("Solubility"):
                        output += f"**Solubility:** {data.get('Solubility')}\n"
                    if data.get("LogP"):
                        output += f"**LogP (Hydrophobicity):** {data.get('LogP')}\n"
                    if data.get("PolarSurfaceArea"):
                        output += f"**Polar Surface Area:** {data.get('PolarSurfaceArea')} Å²\n"

                # Synonyms
                if data.get("Synonyms"):
                    synonyms = str(data.get("Synonyms")).split(" | ")[:10]
                    output += f"\n## Synonyms\n{' | '.join(synonyms)}\n"

                # Image from CAS
                if data.get("Image"):
                    output += f"\n## Structural Formula\n![Compound Structure]({data.get('Image')})\n"

                # Data Sources
                output += "\n## Data Sources\n"
                if self.valves.use_pubchem and result["pubchem_data"]:
                    output += (
                        "✅ **PubChem** - primary chemical properties and identifiers\n"
                    )
                if self.valves.use_cas and result["cas_data"]:
                    cas_sources = [
                        f"CAS {cas_rn}"
                        for cas_rn, d in result["cas_data"].items()
                        if d and isinstance(d, dict)
                    ]
                    if cas_sources:
                        output += f"✅ **{' | '.join(cas_sources)}** - experimental properties and structural formula\n"

                # Translation Information
                output += "\n## Translation Information\n"
                output += f"**Provider:** {result.get('llm_provider', 'openrouter')}\n"
                output += f"**Model Used:** {result.get('used_model', self.valves.openrouter_model)}\n"
                output += f"**API Endpoint:** {result.get('llm_url', _get_openrouter_chat_url(self.valves))}\n"
                output += f"**Original Name:** `{compound_name}` → **English Name:** `{translated_name}`\n"

            else:
                output += "❌ **Data Not Found**\n\n"
                output += "Possible reasons:\n"
                output += "- Incorrect compound name\n"
                output += "- Compound is missing from databases\n"
                output += "- Connection issues with APIs\n\n"
                output += "**Recommendations:**\n"
                output += "- Check spelling of the name\n"
                output += "- Try using the chemical formula or CAS number\n"
                output += "- Ensure the tool settings are correct\n"
                output += f"- Provider: **openrouter**\n"
                output += f"- Model: **{result.get('used_model', self.valves.openrouter_model)}**\n"
                output += f"- API Endpoint: **{result.get('llm_url', _get_openrouter_chat_url(self.valves))}**\n"
                output += f"- Translated Name: `{translated_name}`\n"

            if self.valves.debug_mode:
                debug_info = "\n\n<details>\n"
                debug_info += "### Runtime\n"
                debug_info += f"**Provider:** openrouter\n"
                debug_info += f"**Model:** {result.get('used_model', self.valves.openrouter_model)}\n"
                debug_info += f"**Endpoint:** {_get_openrouter_chat_url(self.valves)}\n"
                debug_info += f"**PubChem Enabled:** {self.valves.use_pubchem}\n"
                debug_info += f"**CAS Enabled:** {self.valves.use_cas}\n"
                if self.valves.use_cas:
                    debug_info += f"**CAS API Key length:** {len(self.valves.cas_api_key) if self.valves.cas_api_key else 0}\n"

                debug_info += "\n### Raw Data from Sources:\n"

                if result["pubchem_data"]:
                    debug_info += "\n**PubChem Raw Data:**\n"
                    debug_info += f"```json\n{json.dumps(result['pubchem_data'], indent=2, ensure_ascii=False)}\n```\n"

                for cas_rn, cas_data in result["cas_data"].items():
                    if cas_data and isinstance(cas_data, dict):
                        debug_info += f"\n**CAS {cas_rn} Raw Data:**\n"
                        debug_info += f"```json\n{json.dumps(cas_data, indent=2, ensure_ascii=False)}\n```\n"

                debug_info += "\n### Debug Logs:\n"
                debug_info += "\n".join(self.debug_logs)
                debug_info += "\n\n</details>"

                output += debug_info

            _log(
                self.debug_logs,
                self.valves,
                f"--- SEARCH COMPLETED: {'SUCCESSFUL' if result['success'] else 'UNSUCCESSFUL'} ---",
            )
            return output

        except Exception as e:
            error_msg = (
                "❌ **CRITICAL ERROR**\n\n"
                f"An unexpected error occurred while searching for compound data '{compound_name}':\n\n"
                f"`{str(e)}`\n\n"
                "Please check the tool settings or try a different request."
            )
            _log(self.debug_logs, self.valves, f"CRITICAL ERROR: {str(e)}")

            if self.valves.debug_mode:
                error_msg += "\n\n<details>\n<summary>Error Details</summary>\n\n"
                error_msg += "```\n" + "\n".join(self.debug_logs[-20:]) + "\n```\n"
                error_msg += f"\n**Provider:** openrouter\n**Model:** {self.valves.openrouter_model}\n"
                error_msg += (
                    f"**Endpoint:** {_get_openrouter_chat_url(self.valves)}\n</details>"
                )

            return error_msg
