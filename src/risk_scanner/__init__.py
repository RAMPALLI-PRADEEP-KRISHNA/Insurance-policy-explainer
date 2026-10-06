"""Risk and red-flag scanning package for insurance policies."""

from src.risk_scanner.red_flag_scanner import (
    RedFlagScanner,
    red_flag_scanner,
    scan_policy,
)

__all__ = [
    "RedFlagScanner",
    "red_flag_scanner",
    "scan_policy",
]
