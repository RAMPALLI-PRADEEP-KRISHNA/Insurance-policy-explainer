"""
Fact normalizer and difference generator for insurance policy comparison.
Extracts normalized numeric values, units, and qualifiers while preserving
original wording, conditions, and strictly avoiding unwarranted conversions.
"""

import re
from typing import Optional, Tuple
from src.models.data_models import (
    ComparisonCategory,
    ComparisonCellStatus,
    ComparisonFact,
)


class ComparisonNormalizer:
    """Normalizes extracted text snippets into structured ComparisonFact attributes."""

    def normalize_duration(self, text: str) -> Tuple[Optional[float], Optional[str], Optional[str], Optional[str]]:
        """
        Extract numeric duration, unit, original string, and condition.
        Returns: (value, unit, original_value, conditions)
        """
        # Match durations like "36 months", "24 months", "30 days", "2 years"
        m = re.search(r"(\d+)\s*(months?|years?|days?)", text, re.IGNORECASE)
        if not m:
            return None, None, None, None

        num = float(m.group(1))
        unit = m.group(2).lower()
        orig = f"{m.group(1)} {m.group(2)}"

        # Standardize units: convert years to months if desired, or keep as months/days
        if "year" in unit:
            num = num * 12.0
            unit = "months"

        # Check conditions
        cond = None
        if re.search(r"continuous\s+coverage", text, re.IGNORECASE):
            cond = "continuous coverage"
        elif re.search(r"specified\s+procedures?|specific\s+illness", text, re.IGNORECASE):
            cond = "for specified procedures"
        elif re.search(r"from\s+inception", text, re.IGNORECASE):
            cond = "from inception"

        return num, unit, orig, cond

    def normalize_percentage(self, text: str) -> Tuple[Optional[float], Optional[str], Optional[str], Optional[str]]:
        """
        Extract percentage value, unit, original string, and qualifiers.
        Returns: (value, unit, original_value, conditions)
        """
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        if not m:
            return None, None, None, None

        val = float(m.group(1))
        orig = f"{m.group(1)}%"

        # Check conditions (e.g. age 61+, per day / Sum Insured)
        cond_parts = []
        age_m = re.search(r"age\s*(\d+)(?:\s*(?:and\s+above|\+))?", text, re.IGNORECASE)
        if age_m:
            cond_parts.append(f"for age {age_m.group(1)} and above")

        if re.search(r"of\s*sum\s*insured", text, re.IGNORECASE):
            cond_parts.append("of Sum Insured")

        if re.search(r"per\s*day", text, re.IGNORECASE):
            cond_parts.append("per day")

        cond = ", ".join(cond_parts) if cond_parts else None
        return val, "percent", orig, cond

    def normalize_currency(self, text: str) -> Tuple[Optional[float], Optional[str], Optional[str], Optional[str]]:
        """
        Extract monetary amount, unit ('INR'), original string, and qualifiers.
        Returns: (value, unit, original_value, conditions)
        """
        # Match amounts like ₹5,000, Rs. 10,000, Rs 5,00,000
        m = re.search(r"(?:rs\.?|₹|inr)\s*([\d,]+)", text, re.IGNORECASE)
        if not m:
            return None, None, None, None

        clean_num_str = m.group(1).replace(",", "")
        try:
            val = float(clean_num_str)
        except ValueError:
            return None, None, None, None

        orig = f"Rs. {m.group(1)}"
        cond_parts = []
        if re.search(r"per\s*day", text, re.IGNORECASE):
            cond_parts.append("per day")
        if re.search(r"per\s*(?:claim|policy\s*year)", text, re.IGNORECASE):
            cond_parts.append("per policy year")

        cond = ", ".join(cond_parts) if cond_parts else None
        return val, "INR", orig, cond


comparison_normalizer = ComparisonNormalizer()


def generate_difference_description(fact_a: ComparisonFact, fact_b: ComparisonFact) -> str:
    """
    Generate an objective, strictly non-judgmental difference description.
    Never states 'Policy A is better', and never claims absent facts when NOT_MENTIONED.
    """
    cat_label = fact_a.category.value.replace("_", " ").lower()

    # Case 1: Both NOT_MENTIONED
    if fact_a.status == ComparisonCellStatus.NOT_MENTIONED and fact_b.status == ComparisonCellStatus.NOT_MENTIONED:
        return f"Neither policy mentions specific terms for {cat_label} in the evaluated sections."

    # Case 2: One NOT_MENTIONED, one FOUND
    if fact_a.status == ComparisonCellStatus.NOT_MENTIONED and fact_b.status == ComparisonCellStatus.FOUND:
        b_val = fact_b.original_value or (f"{fact_b.value} {fact_b.unit}" if fact_b.value is not None else fact_b.description)
        b_cond = f" ({fact_b.conditions})" if fact_b.conditions else ""
        return f"Policy B explicitly states {b_val}{b_cond}; no corresponding {cat_label} clause was identified in Policy A."

    if fact_b.status == ComparisonCellStatus.NOT_MENTIONED and fact_a.status == ComparisonCellStatus.FOUND:
        a_val = fact_a.original_value or (f"{fact_a.value} {fact_a.unit}" if fact_a.value is not None else fact_a.description)
        a_cond = f" ({fact_a.conditions})" if fact_a.conditions else ""
        return f"Policy A explicitly states {a_val}{a_cond}; no corresponding {cat_label} clause was identified in Policy B."

    # Case 3: EXPLICITLY_ABSENT handling
    if fact_a.category == ComparisonCategory.PERMANENT_EXCLUSIONS:
        if fact_a.status == ComparisonCellStatus.FOUND and fact_b.status == ComparisonCellStatus.EXPLICITLY_ABSENT:
            return "Policy A permanently excludes aesthetic surgery; Policy B covers aesthetic surgery subject to conditions."
        if fact_b.status == ComparisonCellStatus.FOUND and fact_a.status == ComparisonCellStatus.EXPLICITLY_ABSENT:
            return "Policy B permanently excludes aesthetic surgery; Policy A covers aesthetic surgery subject to conditions."

    if fact_a.status == ComparisonCellStatus.EXPLICITLY_ABSENT and fact_b.status == ComparisonCellStatus.FOUND:
        b_val = fact_b.original_value or (f"{fact_b.value} {fact_b.unit}" if fact_b.value is not None else fact_b.description)
        b_cond = f" ({fact_b.conditions})" if fact_b.conditions else ""
        return f"Policy A explicitly specifies that no {cat_label} applies (or is not covered); Policy B specifies {b_val}{b_cond}."

    if fact_b.status == ComparisonCellStatus.EXPLICITLY_ABSENT and fact_a.status == ComparisonCellStatus.FOUND:
        a_val = fact_a.original_value or (f"{fact_a.value} {fact_a.unit}" if fact_a.value is not None else fact_a.description)
        a_cond = f" ({fact_a.conditions})" if fact_a.conditions else ""
        return f"Policy B explicitly specifies that no {cat_label} applies (or is not covered); Policy A specifies {a_val}{a_cond}."

    if fact_a.status == ComparisonCellStatus.EXPLICITLY_ABSENT and fact_b.status == ComparisonCellStatus.EXPLICITLY_ABSENT:
        return f"Both policies explicitly state that this provision is absent, not applicable, or not covered."

    # Case 4: AMBIGUOUS handling
    if fact_a.status == ComparisonCellStatus.AMBIGUOUS and fact_b.status == ComparisonCellStatus.AMBIGUOUS:
        return f"Both policies contain ambiguous or conflicting clauses regarding {cat_label}."
    if fact_a.status == ComparisonCellStatus.AMBIGUOUS:
        b_desc = fact_b.original_value or (f"{fact_b.value} {fact_b.unit}" if fact_b.value is not None else fact_b.description)
        return f"Policy A contains ambiguous or conflicting clauses; Policy B specifies {b_desc}."
    if fact_b.status == ComparisonCellStatus.AMBIGUOUS:
        a_desc = fact_a.original_value or (f"{fact_a.value} {fact_a.unit}" if fact_a.value is not None else fact_a.description)
        return f"Policy A specifies {a_desc}; Policy B contains ambiguous or conflicting clauses."

    # Case 5: Both FOUND - Check units and basis
    if fact_a.status == ComparisonCellStatus.FOUND and fact_b.status == ComparisonCellStatus.FOUND:
        # Check if units differ (e.g. percent vs INR)
        if fact_a.unit and fact_b.unit and fact_a.unit != fact_b.unit:
            a_str = fact_a.original_value or f"{fact_a.value} {fact_a.unit}"
            b_str = fact_b.original_value or f"{fact_b.value} {fact_b.unit}"
            a_cond = f" ({fact_a.conditions})" if fact_a.conditions else ""
            b_cond = f" ({fact_b.conditions})" if fact_b.conditions else ""
            return f"Different limit basis: Policy A specifies {a_str}{a_cond}, whereas Policy B specifies {b_str}{b_cond}."

        # Check if conditions differ (e.g. age 61+ vs all ages)
        if fact_a.conditions != fact_b.conditions and (fact_a.conditions or fact_b.conditions):
            a_str = fact_a.original_value or str(fact_a.value)
            b_str = fact_b.original_value or str(fact_b.value)
            a_cond = f" ({fact_a.conditions})" if fact_a.conditions else " (standard)"
            b_cond = f" ({fact_b.conditions})" if fact_b.conditions else " (standard)"
            return f"Conditions differ: Policy A specifies {a_str}{a_cond}; Policy B specifies {b_str}{b_cond}."

        # Same unit, same condition - numeric comparison
        if fact_a.value is not None and fact_b.value is not None:
            try:
                val_a = float(fact_a.value)
                val_b = float(fact_b.value)
                unit_str = f" {fact_a.unit}" if fact_a.unit else ""

                if val_a == val_b:
                    orig_str = fact_a.original_value or f"{val_a}{unit_str}"
                    return f"Both policies state an identical figure of {orig_str}."
                
                # State both values objectively
                if "waiting" in cat_label:
                    if val_b < val_a:
                        return f"Policy B states a {val_b:g}{unit_str} waiting period, compared with {val_a:g}{unit_str} in Policy A."
                    else:
                        return f"Policy A states a {val_a:g}{unit_str} waiting period, compared with {val_b:g}{unit_str} in Policy B."
                else:
                    return f"Policy A states {val_a:g}{unit_str}; Policy B states {val_b:g}{unit_str}."
            except (ValueError, TypeError):
                pass

        # Textual / non-numeric facts
        if fact_a.description == fact_b.description:
            return "Both policies contain comparable terms for this provision."
        return f"Policy A states: '{fact_a.description}'; Policy B states: '{fact_b.description}'."

    return "Policy terms differ as detailed in the policy facts."
