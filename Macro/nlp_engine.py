"""Central-bank statement sentiment analysis for quantitative trading."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, ValidationError, field_validator


FED_FOMC_INDEX = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
ECB_PRESS_INDEX = "https://www.ecb.europa.eu/press/pr/date/html/index.en.html"
USER_AGENT = "MacroNLPResearch/1.0 (central-bank statement analysis)"
REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / \
    "data" / "sentiment_cache.json"

SYSTEM_PROMPT = """You are an expert monetary-policy economist and FX analyst.
Analyze the supplied official statement from {central_bank_name}. Score its
hawkishness versus dovishness from -5.0 to +5.0:
-5.0 is extremely dovish (imminent cuts, growth concerns, inflation under
control); 0.0 is neutral or in line with expectations; +5.0 is extremely
hawkish (hikes or higher-for-longer rates, inflation concerns).

Return exactly one JSON object matching the requested schema. Use only evidence
in the text; do not invent market expectations or facts. Provide exactly three
concise key drivers. The trader summary must contain exactly two sentences
describing the likely FX-market implications. Use the statement's publication
date in YYYY-MM-DD format. The score is positive for hawkish and negative for
dovish. Consider the central bank's own reaction function, not a comparison to
the other central bank."""


class MonetaryPolicyAnalysis(BaseModel):
    """Validated, machine-readable monetary-policy sentiment analysis."""

    central_bank: Literal["FED", "ECB"]
    statement_date: date
    hawk_dove_score: float = Field(ge=-5.0, le=5.0)
    key_drivers: list[str] = Field(min_length=3, max_length=3)
    inflation_stance: Literal["High", "Moderate", "Low"]
    growth_stance: Literal["Weak", "Moderate", "Strong"]
    summary_for_trader: str = Field(min_length=1)

    @field_validator("key_drivers")
    @classmethod
    def drivers_must_not_be_blank(cls, drivers: list[str]) -> list[str]:
        if any(not driver.strip() for driver in drivers):
            raise ValueError("Key drivers must not be blank.")
        return drivers


def _read_sentiment_cache(cache_path: Path) -> dict[str, Any]:
    """Read the versioned sentiment cache, returning an empty cache if absent."""
    if not cache_path.exists():
        return {"version": 1, "analyses": {}}
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("analyses"), dict):
        raise ValueError(f"Invalid sentiment cache structure in {cache_path}.")
    return payload


def load_sentiment_cache(cache_path: str | Path = DEFAULT_CACHE_PATH) -> pd.DataFrame:
    """Load cached FED/ECB scores as a dated event dataframe."""
    path = Path(cache_path)
    cache = _read_sentiment_cache(path)
    rows: list[dict[str, Any]] = []
    for key, entry in cache["analyses"].items():
        if not isinstance(entry, dict) or not isinstance(entry.get("analysis"), dict):
            continue
        try:
            analysis = MonetaryPolicyAnalysis.model_validate(entry["analysis"])
        except ValidationError:
            continue
        rows.append(
            {
                "statement_date": analysis.statement_date,
                "central_bank": analysis.central_bank,
                "hawk_dove_score": analysis.hawk_dove_score,
                "cache_key": key,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["central_bank", "hawk_dove_score"],
            index=pd.DatetimeIndex([], name="Date"),
        )

    scores = pd.DataFrame(rows).drop_duplicates(
        ["statement_date", "central_bank"], keep="last"
    )
    scores["statement_date"] = pd.to_datetime(scores["statement_date"])
    scores = scores.pivot(
        index="statement_date", columns="central_bank", values="hawk_dove_score"
    ).rename(columns={"FED": "FED_Score", "ECB": "ECB_Score"})
    scores.index.name = "Date"
    return scores.sort_index()


def _request(url: str) -> requests.Response:
    """Fetch a URL with a stable user agent and a bounded timeout."""
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response


def _extract_html_text(html: str) -> str:
    """Extract readable document text, excluding page chrome and scripts."""
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        element.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    return "\n".join(
        line.strip() for line in main.get_text("\n").splitlines() if line.strip()
    )


def fetch_document(url: str) -> str:
    """Fetch a statement from a URL, extracting HTML or PDF text."""
    response = _request(url)
    content_type = response.headers.get("Content-Type", "").lower()
    is_pdf = "pdf" in content_type or urlparse(
        url).path.lower().endswith(".pdf")
    if is_pdf:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("Install pypdf to read PDF documents.") from exc
        reader = PdfReader(BytesIO(response.content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        response.encoding = response.encoding or "utf-8"
        text = _extract_html_text(response.text)
    if not text.strip():
        raise ValueError(f"No text could be extracted from {url!r}.")
    return text


def load_document(source: str) -> str:
    """Load text from a URL, TXT/PDF path, or literal text input."""
    if re.match(r"^https?://", source.strip(), flags=re.IGNORECASE):
        return fetch_document(source.strip())

    path = Path(source).expanduser()
    if path.is_file():
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise RuntimeError(
                    "Install pypdf to read PDF documents.") from exc
            text = "\n".join(
                page.extract_text() or "" for page in PdfReader(path).pages
            )
        elif path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8")
        else:
            raise ValueError(
                "Supported local document formats are TXT, MD, and PDF.")
        if not text.strip():
            raise ValueError(f"No text could be extracted from {str(path)!r}.")
        return text

    return source


def _find_latest_official_url(central_bank: Literal["FED", "ECB"]) -> str:
    """Find the most recent matching statement from an official listing page."""
    if central_bank == "FED":
        index_url = FED_FOMC_INDEX
        allowed_host = "federalreserve.gov"
        keywords = ("statement", "minutes")
        reject = ("transcript", "implementation note", "press conference")
    else:
        index_url = ECB_PRESS_INDEX
        allowed_host = "ecb.europa.eu"
        keywords = ("monetary policy decisions", "monetary policy statement")
        reject = ("account", "speech", "interview")

    soup = BeautifulSoup(_request(index_url).text, "html.parser")
    candidates: list[tuple[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        label = " ".join(anchor.get_text(" ", strip=True).lower().split())
        href = urljoin(index_url, str(anchor["href"]))
        combined = f"{label} {href.lower()}"
        host = urlparse(href).hostname or ""
        if not host.endswith(allowed_host):
            continue
        if any(keyword in combined for keyword in keywords) and not any(
            excluded in combined for excluded in reject
        ):
            candidates.append((label, href))

    if not candidates:
        raise RuntimeError(
            f"No matching {central_bank} statement link found at {index_url}. "
            "Pass a statement URL directly instead."
        )
    return candidates[0][1]


def fetch_latest_statement(central_bank: Literal["FED", "ECB"]) -> tuple[str, str]:
    """Fetch the newest matching FED or ECB statement from its official site.

    Returns the selected document URL and its extracted text. FED results come
    from the FOMC calendar listing; ECB results come from the press-release
    listing and match monetary-policy decision/statement links.
    """
    document_url = _find_latest_official_url(central_bank)
    return document_url, fetch_document(document_url)


def analyze_statement(
    text: str,
    central_bank: Literal["FED", "ECB"],
    model: str | None = None,
    document_date: date | str | None = None,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
) -> MonetaryPolicyAnalysis:
    """Classify statement sentiment with a LiteLLM-supported model.

    Set ``LLM_MODEL`` and the provider's API key in the environment, or pass
    ``model`` explicitly. LiteLLM supports OpenAI, Anthropic, and compatible
    open-source model endpoints. If ``document_date`` is supplied, a cached
    analysis for the same central bank and date is returned before any API call.
    """
    if not text.strip():
        raise ValueError("Statement text must not be empty.")
    cache_file = Path(cache_path)
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    requested_date = (
        date.fromisoformat(document_date)
        if isinstance(document_date, str)
        else document_date
    )
    cache = _read_sentiment_cache(cache_file)
    for entry in cache["analyses"].values():
        if not isinstance(entry, dict):
            continue
        if requested_date is None and entry.get("text_sha256") != content_hash:
            continue
        analysis_data = entry.get("analysis")
        if not isinstance(analysis_data, dict):
            continue
        cached = MonetaryPolicyAnalysis.model_validate(analysis_data)
        if cached.central_bank == central_bank and (
            requested_date is None or cached.statement_date == requested_date
        ):
            return cached

    model_name = model or os.getenv("LLM_MODEL")
    if not model_name:
        raise RuntimeError(
            "Set LLM_MODEL (for example, openai/gpt-4o or anthropic/claude-3-5-sonnet)."
        )
    try:
        from litellm import completion
    except ImportError as exc:
        raise RuntimeError("Install litellm to call an LLM provider.") from exc

    central_bank_name = "Federal Reserve" if central_bank == "FED" else "ECB"
    response = completion(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT.format(central_bank_name=central_bank_name),
            },
            {
                "role": "user",
                "content": (
                    f"Central bank: {central_bank}\n"
                    "Return these fields: central_bank, statement_date, "
                    "hawk_dove_score, key_drivers, inflation_stance, "
                    "growth_stance, summary_for_trader.\n\n"
                    f"Statement text:\n{text}"
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("The LLM returned an empty response.")
    try:
        payload: Any = json.loads(content)
        result = MonetaryPolicyAnalysis.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(
            f"The LLM response did not match the required JSON schema: {exc}") from exc
    if result.central_bank != central_bank:
        raise ValueError(
            f"LLM returned central_bank={result.central_bank}, expected {central_bank}."
        )
    if requested_date is not None and result.statement_date != requested_date:
        raise ValueError(
            f"LLM returned statement_date={result.statement_date}, "
            f"expected {requested_date}."
        )

    cache_key = (
        f"{result.central_bank}:{result.statement_date.isoformat()}:{content_hash}"
    )
    cache["analyses"][cache_key] = {
        "text_sha256": content_hash,
        "source_text": text,
        "analysis": result.model_dump(mode="json"),
    }
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
    temporary_file.write_text(
        json.dumps(cache, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    temporary_file.replace(cache_file)
    return result


def analyze_batch(
    sources: list[str],
    central_bank: Literal["FED", "ECB"],
    model: str | None = None,
    statement_dates: list[date | str] | None = None,
    cache_path: str | Path = DEFAULT_CACHE_PATH,
) -> pd.DataFrame:
    """Analyze a batch of text, URL, TXT, or PDF sources into a DataFrame."""
    if statement_dates is not None and len(statement_dates) != len(sources):
        raise ValueError("statement_dates must have one date per source.")
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        analysis = analyze_statement(
            load_document(source),
            central_bank,
            model,
            document_date=(statement_dates[index]
                           if statement_dates else None),
            cache_path=cache_path,
        )
        row = analysis.model_dump(mode="json")
        row["source"] = source
        rows.append(row)
    return pd.DataFrame(rows)


def combine_quant_and_nlp_signals(
    quant_signal: int,
    fed_score: float,
    ecb_score: float,
) -> int:
    """Confirm a quantitative EUR/USD signal with the relative policy tone.

    Returns -1 (short EUR/USD), 0 (flat), or +1 (long EUR/USD).
    """
    if quant_signal not in {-1, 0, 1}:
        raise ValueError("quant_signal must be -1, 0, or 1.")
    if not (-5 <= fed_score <= 5 and -5 <= ecb_score <= 5):
        raise ValueError("FED and ECB scores must each be between -5 and +5.")
    sentiment_diff = fed_score - ecb_score
    if quant_signal == -1 and sentiment_diff >= -1.0:
        return -1
    if quant_signal == 1 and sentiment_diff <= 1.0:
        return 1
    return 0


def run_sample_analysis(
    sample_text: str,
    central_bank: Literal["FED", "ECB"] = "FED",
    quant_signal: int = -1,
    model: str | None = None,
) -> tuple[MonetaryPolicyAnalysis, int]:
    """Analyze sample text and return its validated output and fused signal."""
    analysis = analyze_statement(sample_text, central_bank, model)
    if central_bank == "FED":
        fed_score, ecb_score = analysis.hawk_dove_score, 0.0
    else:
        fed_score, ecb_score = 0.0, analysis.hawk_dove_score
    final_signal = combine_quant_and_nlp_signals(
        quant_signal, fed_score, ecb_score
    )
    return analysis, final_signal


SAMPLE_STATEMENT = """The Committee decided to maintain the target range for the
federal funds rate. Inflation has made progress but remains somewhat elevated.
The Committee remains strongly committed to returning inflation to its 2 percent
objective and will assess incoming data, the evolving outlook, and the balance
of risks before considering any adjustments to the target range."""


def main() -> None:
    """Run a sample analysis or analyze a supplied document from the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", help="Statement text, URL, or TXT/PDF file path.")
    parser.add_argument("--latest", choices=("FED", "ECB"),
                        help="Fetch latest official statement.")
    parser.add_argument(
        "--central-bank", choices=("FED", "ECB"), default="FED")
    parser.add_argument("--quant-signal", type=int,
                        choices=(-1, 0, 1), default=-1)
    parser.add_argument(
        "--model", help="LiteLLM model identifier; defaults to LLM_MODEL.")
    parser.add_argument(
        "--statement-date", help="Publication date (YYYY-MM-DD) for cache lookup.")
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    args = parser.parse_args()

    central_bank = args.latest or args.central_bank
    source = args.source
    if args.latest:
        source, text = fetch_latest_statement(args.latest)
        print(f"Source URL: {source}")
    else:
        text = load_document(source) if source else SAMPLE_STATEMENT

    analysis = analyze_statement(
        text,
        central_bank,
        args.model,
        document_date=args.statement_date,
        cache_path=args.cache_path,
    )
    if central_bank == "FED":
        fed_score, ecb_score = analysis.hawk_dove_score, 0.0
    else:
        fed_score, ecb_score = 0.0, analysis.hawk_dove_score
    final_signal = combine_quant_and_nlp_signals(
        args.quant_signal, fed_score, ecb_score
    )

    print("\nValidated LLM analysis")
    print(analysis.model_dump_json(indent=2))
    print(f"\nQuant signal: {args.quant_signal:+d}")
    print(f"Sentiment difference (FED - ECB): {fed_score - ecb_score:+.2f}")
    print(
        f"Final EUR/USD signal: {final_signal:+d} (-1 short, 0 flat, +1 long)")


if __name__ == "__main__":
    main()
