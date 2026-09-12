r"""
Author: Brian Lowe
Copyright (c) 2026 Brian Lowe. All rights reserved.
File: risk/walk_away.py
Date of Creation: September 9, 2026
Path: Xanthommatin/risk/walk_away.py

Annotations:
- Implementation of 'The Walk-Away Matrix' executive snapshot and strategic ablation model.
- Calculates Net Terminal Position (NTLV) and real-time operational burn runway metrics.
- Provides structural ablation modeling to stress-test business overhead and cost centers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class WalkAwaySnapshot:
    cash_in_bank: float
    immediate_receivables: float
    accounts_payable: float
    wind_down_costs: float
    monthly_burn: float

    @property
    def total_assets(self) -> float:
        return self.cash_in_bank + self.immediate_receivables

    @property
    def total_liabilities(self) -> float:
        return self.accounts_payable + self.wind_down_costs

    @property
    def net_terminal_position(self) -> float:
        return self.total_assets - self.total_liabilities

    @property
    def current_runway_months(self) -> float:
        if self.monthly_burn <= 0:
            return float("inf")
        return self.cash_in_bank / self.monthly_burn

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assets": {
                "cash_in_bank": self.cash_in_bank,
                "immediate_receivables": self.immediate_receivables,
                "total_assets": self.total_assets,
            },
            "liabilities": {
                "accounts_payable": self.accounts_payable,
                "wind_down_costs": self.wind_down_costs,
                "total_liabilities": self.total_liabilities,
            },
            "metrics": {
                "net_terminal_position": self.net_terminal_position,
                "monthly_burn": self.monthly_burn,
                "current_runway_months": round(self.current_runway_months, 2),
            },
        }


def evaluate_walk_away(snapshot: WalkAwaySnapshot) -> Dict[str, Any]:
    """Evaluate real-time executive exit conditions (The Walk-Away Matrix)."""
    data = snapshot.to_dict()
    ntp = data["metrics"]["net_terminal_position"]

    if ntp > 0:
        status = "SOLVENT_EXIT"
        directive = "Positive net terminal position. Executing walk-away preserves remaining capital without insolvency risk."
    else:
        status = "DEFICIT_EXIT"
        directive = "Negative net terminal position. Immediate capital injection or liability renegotiation required before shutdown."

    data["evaluation"] = {
        "status": status,
        "directive": directive,
    }
    return data


def simulate_structural_ablation(
    snapshot: WalkAwaySnapshot, reduced_burn: float, reduced_wind_down: float
) -> Dict[str, Any]:
    """
    Perform strategic ablation: simulate how cutting a cost center or overhead 
    impacts monthly burn, wind-down exposure, and net terminal position.
    """
    ablated_snapshot = WalkAwaySnapshot(
        cash_in_bank=snapshot.cash_in_bank,
        immediate_receivables=snapshot.immediate_receivables,
        accounts_payable=snapshot.accounts_payable,
        wind_down_costs=max(0.0, snapshot.wind_down_costs - reduced_wind_down),
        monthly_burn=max(0.0, snapshot.monthly_burn - reduced_burn),
    )
    return evaluate_walk_away(ablated_snapshot)


if __name__ == "__main__":
    baseline = WalkAwaySnapshot(
        cash_in_bank=120000.0,
        immediate_receivables=15000.0,
        accounts_payable=25000.0,
        wind_down_costs=20000.0,
        monthly_burn=18000.0,
    )
    print("=== Walk-Away Matrix Baseline Snapshot ===")
    print(json.dumps(evaluate_walk_away(baseline), indent=2))
