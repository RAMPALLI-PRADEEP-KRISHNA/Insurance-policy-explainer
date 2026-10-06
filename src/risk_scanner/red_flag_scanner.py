"""
Evidence-grounded Red-Flag Scanner for insurance policies.
Identifies financially and clinically significant policy restrictions, limitations,
and conditions strictly grounded in extracted document chunks and tables.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone

from src.ingestion.chunker import chunk_document
from src.ingestion.index_store import index_store, IndexStore
from src.models.data_models import (
    Citation,
    Chunk,
    ChunkType,
    Document,
    RedFlag,
    RedFlagReport,
    RiskCategory,
    RiskFinding,
    RiskSeverity,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Heuristic pattern definitions & extraction rules
# ---------------------------------------------------------------------------

PED_PATTERN = re.compile(
    r"(?:pre[- ]?existing|ped).*(?:waiting\s+period|continuous\s+coverage|\bwait\b).*?(?P<dur>\d+\s*(?:months?|years?|days?))|"
    r"(?:waiting\s+period).*(?:pre[- ]?existing|ped).*?(?P<dur2>\d+\s*(?:months?|years?|days?))|"
    r"(?:pre[- ]?existing\s+disease\s+waiting\s+period)",
    re.IGNORECASE,
)

SPECIFIC_DISEASE_WAIT_PATTERN = re.compile(
    r"(?:specific|specified).*(?:illness|disease|procedure|condition|surgery).*waiting\s+period.*?(?P<dur>\d+\s*(?:months?|years?|days?))|"
    r"\b(?:24|twenty[- ]four|two)\s*(?:months?|years?)\s*(?:waiting\s+period|for\s+specified)",
    re.IGNORECASE,
)

INITIAL_WAIT_PATTERN = re.compile(
    r"(?:initial\s+waiting\s+period|30\s+days?\s*(?:from\s+inception|waiting\s+period)|waiting\s+period\s+of\s+30\s+days)",
    re.IGNORECASE,
)

GENERAL_WAIT_PATTERN = re.compile(
    r"(?P<dur>\d+\s*(?:months?|years?|days?))\s*(?:waiting\s+period|of\s+continuous\s+coverage)|"
    r"\bwaiting\s+period\b",
    re.IGNORECASE,
)

PERMANENT_EXCLUSION_PATTERN = re.compile(
    r"\b(?:permanent\s+exclusions?|excluded\s+permanently|shall\s+not\s+be\s+liable|not\s+covered\s+under\s+any\s+circumstances)\b|"
    r"\b(?:hazardous\s+(?:sports|activities)|self[- ]inflicted|aesthetic\s+surgery|cosmetic\s+surgery|substance\s+abuse)\b",
    re.IGNORECASE,
)

SPECIFIC_EXCLUSION_PATTERN = re.compile(
    r"\b(?:specific\s+exclusions?|excluded\s+until|investigation\s*(?:&|and)\s*evaluation|rest\s+cure|dietary\s+supplements?|obesity\s*/\s*weight\s+control|change[- ]of[- ]gender)\b",
    re.IGNORECASE,
)

COPAY_PATTERN = re.compile(
    r"(?:(?P<pct>\d{1,2})\s*%\s*(?:co[- ]?pay(?:ment)?))|"
    r"(?:co[- ]?pay(?:ment)?\s*(?:of\s*)?(?P<pct2>\d{1,2})\s*%)|"
    r"\b(?:co[- ]?payment|co[- ]?pay)\b",
    re.IGNORECASE,
)

ROOM_RENT_PATTERN = re.compile(
    r"\b(?:room\s+rent|room\s+nursing\s+charges|single\s+standard\s+room|single\s+private\s+room|icu\s+(?:charges|rent|limit))\b",
    re.IGNORECASE,
)

DEDUCTIBLE_PATTERN = re.compile(
    r"\b(?:deductibles?|voluntary\s+deductible|aggregate\s+deductible|compulsory\s+deductible)\b",
    re.IGNORECASE,
)

SUBLIMIT_PATTERN = re.compile(
    r"\b(?:sub[- ]?limits?|capped\s+at|maximum\s+payable\s+(?:up\s+to|limit)|cataract\s+limit|limit\s+per\s+surgery|modern\s+treatments?\s+sub[- ]?limit)\b",
    re.IGNORECASE,
)

AGE_RESTRICTION_PATTERN = re.compile(
    r"\b(?:entry\s+age|maximum\s+renewal\s+age|age\s+limit|child\s+covered\s+up\s+to|age\s+\d+\s+and\s+above|senior\s+citizen\s+co[- ]?pay|attained\s+age)\b",
    re.IGNORECASE,
)

PROCEDURE_LIMIT_PATTERN = re.compile(
    r"\b(?:ayush\s+(?:treatment|coverage)?\s*(?:limit|capped)?|organ\s+donor\s+expenses|domiciliary\s+hospitalization\s+limit|bariatric\s+surgery\s+limit|maternity\s+(?:benefit|limit|expenses))\b",
    re.IGNORECASE,
)

CLAIM_CONDITION_PATTERN = re.compile(
    r"\b(?:claim\s+intimation|notice\s+of\s+claim|within\s+(?:\d+|24|48|72)\s+hours|submission\s+of\s+documents\s+within|pre[- ]?authorization|cashless\s+request)\b",
    re.IGNORECASE,
)

RESTORATION_PATTERN = re.compile(
    r"\b(?:restoration|reinstatement|refill)\b.*?(?:unrelated\s+illness|same\s+illness|once\s+in\s+a\s+(?:policy\s+)?year|exhausted)",
    re.IGNORECASE,
)

OTHER_RESTRICTION_PATTERN = re.compile(
    r"\b(?:zone[- ]based\s+co[- ]?pay|geographic\s+restriction|cancellation\s+clause|free[- ]look\s+period|grace\s+period)\b",
    re.IGNORECASE,
)


class RedFlagScanner:
    """
    Analyzes policy chunks and tabular data to identify and prioritize
    clauses that require user review.

    Guarantees:
    - 100% Evidence-first: all detected flags link directly to specific chunk IDs and page numbers.
    - No legal conclusions or subjective disparagement of the policy.
    - Deterministic severity ratings reflecting review priority.
    """

    def __init__(self, index_store_instance: Optional[IndexStore] = None):
        self.index_store = index_store_instance or index_store

    def scan(
        self,
        document: Document,
        chunks: Optional[List[Chunk]] = None,
    ) -> RedFlagReport:
        """
        Scan a document and return a comprehensive RedFlagReport.

        Args:
            document: Document data model.
            chunks: Optional list of chunks. If omitted, loaded from index_store or chunker.

        Returns:
            RedFlagReport containing all detected findings and citations.
        """
        resolved_chunks = self._resolve_chunks(document, chunks)
        if not resolved_chunks:
            logger.warning("No chunks available to scan for doc_id=%s", document.doc_id)
            return RedFlagReport(
                document_id=document.doc_id,
                filename=document.filename,
                total_flags=0,
                findings=[],
                categories_found=[],
            )

        findings: List[RedFlag] = []
        seen_keys: Set[Tuple[RiskCategory, str, str]] = set()

        for chunk in resolved_chunks:
            # We inspect both full chunk text and individual table rows
            text_units = self._extract_text_units(chunk)
            for unit_text in text_units:
                detected_flags = self._evaluate_text_unit(document, chunk, unit_text)
                for flag in detected_flags:
                    dedup_key = (flag.category, flag.chunk_id, flag.title)
                    if dedup_key not in seen_keys:
                        seen_keys.add(dedup_key)
                        findings.append(flag)

        # Sort findings: HIGH first, then MEDIUM, then LOW, then page_number
        severity_rank = {RiskSeverity.HIGH: 0, RiskSeverity.MEDIUM: 1, RiskSeverity.LOW: 2}
        findings.sort(key=lambda f: (severity_rank.get(f.severity, 3), f.page_number, f.title))

        categories_found = sorted(
            list({f.category for f in findings}),
            key=lambda c: c.value,
        )

        return RedFlagReport(
            document_id=document.doc_id,
            filename=document.filename,
            total_flags=len(findings),
            findings=findings,
            categories_found=categories_found,
            scanned_at=datetime.now(timezone.utc),
        )

    def scan_risk_findings(
        self,
        document: Document,
        chunks: Optional[List[Chunk]] = None,
    ) -> List[RiskFinding]:
        """
        Generate category-level RiskFinding objects conforming to DATA_MODEL.md Section 8.

        Args:
            document: Ingested policy Document.
            chunks: Optional list of chunks.

        Returns:
            List of RiskFinding objects covering all canonical risk categories.
        """
        report = self.scan(document, chunks)
        resolved_chunks = self._resolve_chunks(document, chunks)
        chunk_map = {c.chunk_id: c for c in resolved_chunks}

        # Categories to report
        categories_to_evaluate = [
            RiskCategory.WAITING_PERIOD,
            RiskCategory.PERMANENT_EXCLUSION,
            RiskCategory.SPECIFIC_EXCLUSION,
            RiskCategory.CO_PAYMENT,
            RiskCategory.ROOM_RENT_LIMIT,
            RiskCategory.SUB_LIMIT,
            RiskCategory.DEDUCTIBLE,
            RiskCategory.AGE_RESTRICTION,
            RiskCategory.DISEASE_OR_PROCEDURE_LIMIT,
            RiskCategory.CLAIM_CONDITION,
            RiskCategory.RESTORATION_RESTRICTION,
            RiskCategory.OTHER_MATERIAL_RESTRICTION,
        ]

        results: List[RiskFinding] = []
        for cat in categories_to_evaluate:
            cat_flags = [f for f in report.findings if f.category == cat]
            if cat_flags:
                # Deduplicate source chunks & citations
                src_chunks = []
                for f in cat_flags:
                    if f.chunk_id in chunk_map and chunk_map[f.chunk_id] not in src_chunks:
                        src_chunks.append(chunk_map[f.chunk_id])

                all_citations = []
                for f in cat_flags:
                    all_citations.extend(f.citations)

                # Highest severity among flags
                severities = [f.severity for f in cat_flags]
                if RiskSeverity.HIGH in severities:
                    max_sev = RiskSeverity.HIGH
                elif RiskSeverity.MEDIUM in severities:
                    max_sev = RiskSeverity.MEDIUM
                else:
                    max_sev = RiskSeverity.LOW

                summary = "; ".join([f.title for f in cat_flags[:3]])
                if len(cat_flags) > 3:
                    summary += f" (and {len(cat_flags) - 3} more)"

                results.append(
                    RiskFinding(
                        doc_id=document.doc_id,
                        category=cat,
                        found=True,
                        summary=summary,
                        source_chunks=src_chunks,
                        citations=all_citations,
                        severity=max_sev,
                    )
                )
            else:
                cat_label = cat.value.lower().replace("_", " ")
                results.append(
                    RiskFinding(
                        doc_id=document.doc_id,
                        category=cat,
                        found=False,
                        summary=f"No explicit {cat_label} clauses detected in policy.",
                        source_chunks=[],
                        citations=[],
                        severity=None,
                    )
                )

        return results

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    def _resolve_chunks(
        self,
        document: Document,
        chunks: Optional[List[Chunk]],
    ) -> List[Chunk]:
        """Resolve chunk list from provided argument, index store, or chunker."""
        if chunks is not None:
            return chunks

        if self.index_store.index_exists(document.doc_id):
            try:
                _, loaded_chunks, _ = self.index_store.load_index(document.doc_id)
                if loaded_chunks:
                    return loaded_chunks
            except Exception as e:
                logger.warning("Failed to load chunks from index_store: %s", e)

        return chunk_document(document)

    def _extract_text_units(self, chunk: Chunk) -> List[str]:
        """Extract individual lines/rows if table or list, else full chunk text."""
        if chunk.chunk_type == ChunkType.TABLE:
            lines = [line.strip() for line in chunk.text.split("\n") if line.strip()]
            data_rows = []
            is_first_row = True
            for line in lines:
                if re.match(r"^\|?\s*[-:]+\s*\|", line):
                    continue
                if is_first_row:
                    is_first_row = False
                    continue
                data_rows.append(line)
            return data_rows if data_rows else [chunk.text]

        # For standard text/clause chunks:
        # Strip leading breadcrumb "[Section ...]" to avoid duplicate pseudo-matches
        text = chunk.text
        breadcrumb_match = re.match(r"^\[.*?\]\s*\n?", text)
        if breadcrumb_match:
            text = text[breadcrumb_match.end():]

        # Split on multiple numbered sub-clauses if any (e.g., 2.1, 2.2)
        sub_items = re.split(r"\n(?=\d+\.\d+|\b[A-Z]\.|\d+\))\s*", text)
        cleaned = [item.strip() for item in sub_items if item.strip()]
        return cleaned if cleaned else [chunk.text]

    def _make_citation(self, document: Document, chunk: Chunk, snippet: str) -> Citation:
        """Create an authoritative Citation strictly from chunk metadata."""
        clean_snippet = snippet.strip().replace("\n", " ")
        if len(clean_snippet) > 250:
            clean_snippet = clean_snippet[:247] + "..."

        return Citation(
            document_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            filename=document.filename,
            page_number=chunk.page_number,
            page_numbers=chunk.page_numbers if chunk.page_numbers else [chunk.page_number],
            section=chunk.section,
            heading=chunk.subsection or chunk.section,
            source_snippet=clean_snippet,
        )

    def _evaluate_text_unit(
        self,
        document: Document,
        chunk: Chunk,
        text: str,
    ) -> List[RedFlag]:
        """Evaluate a specific string unit against risk categories."""
        flags: List[RedFlag] = []

        # 1. WAITING PERIODS
        # Check PED
        ped_m = PED_PATTERN.search(text)
        if ped_m:
            dur = ped_m.group("dur") or ped_m.group("dur2") or "Specified Duration"
            sev = RiskSeverity.HIGH if any(x in text.lower() for x in ["36", "48", "3 year", "4 year"]) else RiskSeverity.MEDIUM
            title = f"Pre-Existing Disease (PED) Waiting Period ({dur.strip()})" if dur != "Specified Duration" else "Pre-Existing Disease (PED) Waiting Period"
            desc = f"Coverage for declared pre-existing medical conditions requires continuous coverage of {dur.strip()} before claims are admissible."
            flags.append(
                RedFlag(
                    category=RiskCategory.WAITING_PERIOD,
                    title=title,
                    description=desc,
                    severity=sev,
                    document_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    page_numbers=chunk.page_numbers if chunk.page_numbers else [chunk.page_number],
                    section=chunk.section,
                    heading=chunk.subsection or chunk.section,
                    source_snippet=text,
                    citations=[self._make_citation(document, chunk, text)],
                )
            )

        # Check Specific Disease Waiting Period
        spec_m = SPECIFIC_DISEASE_WAIT_PATTERN.search(text)
        if spec_m and not ped_m:
            dur = spec_m.group("dur") or "24 months"
            flags.append(
                RedFlag(
                    category=RiskCategory.WAITING_PERIOD,
                    title=f"Specific Disease Waiting Period ({dur.strip()})",
                    description=f"Specified medical procedures and conditions are subject to a {dur.strip()} waiting period from inception.",
                    severity=RiskSeverity.MEDIUM,
                    document_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    page_numbers=chunk.page_numbers if chunk.page_numbers else [chunk.page_number],
                    section=chunk.section,
                    heading=chunk.subsection or chunk.section,
                    source_snippet=text,
                    citations=[self._make_citation(document, chunk, text)],
                )
            )

        # Check Initial Waiting Period
        init_m = INITIAL_WAIT_PATTERN.search(text)
        if init_m and not ped_m and not spec_m:
            flags.append(
                RedFlag(
                    category=RiskCategory.WAITING_PERIOD,
                    title="Initial Waiting Period (30 Days)",
                    description="Standard 30-day initial waiting period applies from policy inception, during which illnesses (excluding accidents) are not covered.",
                    severity=RiskSeverity.LOW,
                    document_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    page_numbers=chunk.page_numbers if chunk.page_numbers else [chunk.page_number],
                    section=chunk.section,
                    heading=chunk.subsection or chunk.section,
                    source_snippet=text,
                    citations=[self._make_citation(document, chunk, text)],
                )
            )

        # Check General Waiting Period if none matched above
        if not ped_m and not spec_m and not init_m:
            gen_m = GENERAL_WAIT_PATTERN.search(text)
            if gen_m and "no waiting" not in text.lower():
                dur = gen_m.group("dur")
                dur_str = f" ({dur.strip()})" if dur else ""
                flags.append(
                    RedFlag(
                        category=RiskCategory.WAITING_PERIOD,
                        title=f"Waiting Period Condition{dur_str}",
                        description="Policy terms specify a waiting period condition before coverage becomes fully effective.",
                        severity=RiskSeverity.MEDIUM,
                        document_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        page_number=chunk.page_number,
                        page_numbers=chunk.page_numbers if chunk.page_numbers else [chunk.page_number],
                        section=chunk.section,
                        heading=chunk.subsection or chunk.section,
                        source_snippet=text,
                        citations=[self._make_citation(document, chunk, text)],
                    )
                )

        # 2. PERMANENT EXCLUSIONS
        if PERMANENT_EXCLUSION_PATTERN.search(text):
            if "no permanent exclusion" not in text.lower():
                flags.append(
                    RedFlag(
                        category=RiskCategory.PERMANENT_EXCLUSION,
                        title="Permanent Exclusions Clause",
                        description="Lists conditions and treatments permanently excluded from coverage under all circumstances.",
                        severity=RiskSeverity.HIGH,
                        document_id=chunk.doc_id,
                        chunk_id=chunk.chunk_id,
                        page_number=chunk.page_number,
                        page_numbers=chunk.page_numbers if chunk.page_numbers else [chunk.page_number],
                        section=chunk.section,
                        heading=chunk.subsection or chunk.section,
                        source_snippet=text,
                        citations=[self._make_citation(document, chunk, text)],
                    )
                )

        # 3. SPECIFIC EXCLUSIONS
        if SPECIFIC_EXCLUSION_PATTERN.search(text):
            flags.append(
                RedFlag(
                    category=RiskCategory.SPECIFIC_EXCLUSION,
                    title="Specific Treatment / Procedure Exclusion",
                    description="Identifies specific treatments or medical categories excluded or conditional under policy terms.",
                    severity=RiskSeverity.MEDIUM,
                    document_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    page_numbers=chunk.page_numbers if chunk.page_numbers else [chunk.page_number],
                    section=chunk.section,
                    heading=chunk.subsection or chunk.section,
                    source_snippet=text,
                    citations=[self._make_citation(document, chunk, text)],
                )
            )

        # 4. CO-PAYMENT
        copay_m = COPAY_PATTERN.search(text)
        if copay_m and not re.search(r"\b(?:no|zero|nil|0%)\s*co[- ]?pay", text, re.IGNORECASE):
            pct_m = re.search(r"(\d{1,2})\s*%", text)
            pct_num = int(pct_m.group(1)) if pct_m else None
            
            if pct_num is not None:
                if pct_num >= 20:
                    sev = RiskSeverity.HIGH
                elif pct_num >= 10:
                    sev = RiskSeverity.MEDIUM
                else:
                    sev = RiskSeverity.LOW
                pct_label = f" ({pct_num}%)"
            else:
                sev = RiskSeverity.MEDIUM
                pct_label = ""

            age_m = re.search(r"age\s*(\d+)", text, re.IGNORECASE)
            title = f"Co-Payment Requirement{pct_label}"
            if age_m:
                desc = f"A mandatory co-payment of {pct_num if pct_num else 'specified'}% applies for insured persons aged {age_m.group(1)} and above."
            else:
                desc = f"A cost-sharing co-payment{pct_label} requires the insured to personally pay a portion of each admissible claim."

            flags.append(
                RedFlag(
                    category=RiskCategory.CO_PAYMENT,
                    title=title,
                    description=desc,
                    severity=sev,
                    document_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    page_numbers=chunk.page_numbers if chunk.page_numbers else [chunk.page_number],
                    section=chunk.section,
                    heading=chunk.subsection or chunk.section,
                    source_snippet=text,
                    citations=[self._make_citation(document, chunk, text)],
                )
            )

        # 5. ROOM RENT LIMIT
        room_m = ROOM_RENT_PATTERN.search(text)
        if room_m and not re.search(r"\b(?:no\s+room\s+rent\s+cap|no\s+sub[- ]?limit\s+on\s+room|any\s+room\s+category)\b", text, re.IGNORECASE):
            # Extract specific cap if present
            cap_m = re.search(r"(\d+%\s*of\s*(?:sum\s*insured|si)|\d+%\s*per\s*day|(?:rs\.?|₹|inr)\s*[\d,]+)", text, re.IGNORECASE)
            cap_str = f" ({cap_m.group(1).strip()})" if cap_m else ""
            desc = (
                f"Room rent is capped{cap_str}. Exceeding the specified room rent limit can trigger proportionate "
                "deductions across associated medical and surgical expenses on the final hospital bill."
            )
            flags.append(
                RedFlag(
                    category=RiskCategory.ROOM_RENT_LIMIT,
                    title=f"Room Rent Capping Limit{cap_str}",
                    description=desc,
                    severity=RiskSeverity.HIGH,  # Critical financial impact due to proportionate deduction
                    document_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    page_numbers=chunk.page_numbers if chunk.page_numbers else [chunk.page_number],
                    section=chunk.section,
                    heading=chunk.subsection or chunk.section,
                    source_snippet=text,
                    citations=[self._make_citation(document, chunk, text)],
                )
            )

        # 6. DEDUCTIBLE
        ded_m = DEDUCTIBLE_PATTERN.search(text)
        if ded_m and not re.search(r"\b(?:no|nil|zero)\s*deductible\b", text, re.IGNORECASE):
            flags.append(
                RedFlag(
                    category=RiskCategory.DEDUCTIBLE,
                    title="Deductible Requirement",
                    description="Requires policyholder to pay a defined deductible amount out-of-pocket before insurance reimbursement begins.",
                    severity=RiskSeverity.MEDIUM,
                    document_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    page_numbers=chunk.page_numbers if chunk.page_numbers else [chunk.page_number],
                    section=chunk.section,
                    heading=chunk.subsection or chunk.section,
                    source_snippet=text,
                    citations=[self._make_citation(document, chunk, text)],
                )
            )

        # 7. SUB-LIMITS
        sub_m = SUBLIMIT_PATTERN.search(text)
        if sub_m and not room_m:
            flags.append(
                RedFlag(
                    category=RiskCategory.SUB_LIMIT,
                    title="Procedure / Treatment Sub-Limit",
                    description="Limits reimbursement for specific treatments or surgeries to a fixed monetary ceiling regardless of total Sum Insured.",
                    severity=RiskSeverity.MEDIUM,
                    document_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    page_numbers=chunk.page_numbers if chunk.page_numbers else [chunk.page_number],
                    section=chunk.section,
                    heading=chunk.subsection or chunk.section,
                    source_snippet=text,
                    citations=[self._make_citation(document, chunk, text)],
                )
            )

        # 8. AGE RESTRICTION
        age_m = AGE_RESTRICTION_PATTERN.search(text)
        if age_m:
            if re.search(r"\bentry\s+age\b", text, re.IGNORECASE):
                range_m = re.search(r"entry\s+age(?:\s*:\s*|\s+is\s+)(\d+\s*(?:years?)?\s*to\s*\d+\s*(?:years?)?)", text, re.IGNORECASE)
                range_str = f" ({range_m.group(1).strip()})" if range_m else ""
                title = f"Entry Age Limitation{range_str}"
                desc = "Defines the minimum and maximum entry age eligibility criteria for coverage under the policy."
            elif re.search(r"\bmaximum\s+renewal\s+age\b", text, re.IGNORECASE):
                title = "Maximum Renewal Age Restriction"
                desc = "Specifies age limits governing policy renewal and continuation."
            elif re.search(r"age\s*(\d+)\s*(?:and\s+above|\+)", text, re.IGNORECASE):
                age_num = re.search(r"age\s*(\d+)", text, re.IGNORECASE).group(1)
                title = f"Age-Linked Condition (Age {age_num}+)"
                desc = f"Specific policy conditions, co-pays, or limitations trigger when the insured reaches age {age_num}."
            else:
                title = "Age-Related Eligibility / Condition"
                desc = "Specifies entry age limits, renewal ceilings, or age-dependent terms governing eligibility or claims."

            flags.append(
                RedFlag(
                    category=RiskCategory.AGE_RESTRICTION,
                    title=title,
                    description=desc,
                    severity=RiskSeverity.MEDIUM,
                    document_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    page_numbers=chunk.page_numbers if chunk.page_numbers else [chunk.page_number],
                    section=chunk.section,
                    heading=chunk.subsection or chunk.section,
                    source_snippet=text,
                    citations=[self._make_citation(document, chunk, text)],
                )
            )

        # 9. DISEASE / PROCEDURE LIMITS
        proc_m = PROCEDURE_LIMIT_PATTERN.search(text)
        if proc_m and not sub_m:
            flags.append(
                RedFlag(
                    category=RiskCategory.DISEASE_OR_PROCEDURE_LIMIT,
                    title="Specific Treatment / Allowance Cap",
                    description="Imposes special conditions or financial caps on specific medical treatments (e.g. AYUSH, maternity, domiciliary care).",
                    severity=RiskSeverity.MEDIUM,
                    document_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    page_numbers=chunk.page_numbers if chunk.page_numbers else [chunk.page_number],
                    section=chunk.section,
                    heading=chunk.subsection or chunk.section,
                    source_snippet=text,
                    citations=[self._make_citation(document, chunk, text)],
                )
            )

        # 10. CLAIM CONDITIONS
        claim_m = CLAIM_CONDITION_PATTERN.search(text)
        if claim_m:
            flags.append(
                RedFlag(
                    category=RiskCategory.CLAIM_CONDITION,
                    title="Claim Intimation & Procedural Timeline",
                    description="Requires mandatory notification and document submission within strict timeframes for claim approval.",
                    severity=RiskSeverity.LOW,
                    document_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    page_numbers=chunk.page_numbers if chunk.page_numbers else [chunk.page_number],
                    section=chunk.section,
                    heading=chunk.subsection or chunk.section,
                    source_snippet=text,
                    citations=[self._make_citation(document, chunk, text)],
                )
            )

        # 11. RESTORATION RESTRICTIONS
        resto_m = RESTORATION_PATTERN.search(text)
        if resto_m:
            flags.append(
                RedFlag(
                    category=RiskCategory.RESTORATION_RESTRICTION,
                    title="Restoration Benefit Restrictions",
                    description="Restoration of sum insured is conditional, applying only to unrelated illnesses or subject to single-year usage limits.",
                    severity=RiskSeverity.MEDIUM,
                    document_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    page_numbers=chunk.page_numbers if chunk.page_numbers else [chunk.page_number],
                    section=chunk.section,
                    heading=chunk.subsection or chunk.section,
                    source_snippet=text,
                    citations=[self._make_citation(document, chunk, text)],
                )
            )

        # 12. OTHER MATERIAL RESTRICTIONS
        other_m = OTHER_RESTRICTION_PATTERN.search(text)
        if other_m:
            flags.append(
                RedFlag(
                    category=RiskCategory.OTHER_MATERIAL_RESTRICTION,
                    title="Geographic or Policy Condition Clause",
                    description="Outlines territorial, zonal, cancellation, or operational conditions governing policy administration.",
                    severity=RiskSeverity.LOW,
                    document_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    page_numbers=chunk.page_numbers if chunk.page_numbers else [chunk.page_number],
                    section=chunk.section,
                    heading=chunk.subsection or chunk.section,
                    source_snippet=text,
                    citations=[self._make_citation(document, chunk, text)],
                )
            )

        return flags


# Singleton instance
red_flag_scanner = RedFlagScanner()


def scan_policy(
    document: Document,
    chunks: Optional[List[Chunk]] = None,
    scanner_instance: Optional[RedFlagScanner] = None,
) -> RedFlagReport:
    """Convenience function to scan a document using the singleton scanner."""
    scanner = scanner_instance or red_flag_scanner
    return scanner.scan(document, chunks)
