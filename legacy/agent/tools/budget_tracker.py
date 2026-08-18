"""
tools/budget_tracker.py — In-memory trip budget tracker.

Tracks spend per category, warns when approaching or exceeding budget,
and shows percentage used.

Public interface:
    run(action, amount, category, description) -> dict
    reset()       — full reset (called by /reset in CLI)
    get_status()  — direct read without going through run()

Actions:
    set_total   set the overall trip budget
    add/deduct  record a spend
    status      current totals + breakdown + warnings
    reset       clear spends (keeps total)
"""

SCHEMA = {
    "name": "budget_tracker",
    "description": (
        "Manage the trip budget. Track spend by category, check remaining budget, "
        "and get warnings when approaching the limit. "
        "Always call this when a booking is confirmed or a cost is mentioned."
    ),
    "parameters": {
        "action":      "string — set_total | add | deduct | status | reset",
        "amount":      "number — USD amount (required for set_total / add / deduct)",
        "category":    "string — flights | hotel | food | activities | transport | misc",
        "description": "string — short note, e.g. 'Hotel Paris 3 nights'",
    },
}

_WARN_PCT  = 80   # warn when spent >= 80% of total
_OVER_PCT  = 100  # error when spent > 100%


class BudgetTracker:

    def __init__(self) -> None:
        self.total: float      = 0.0
        self.spent: float      = 0.0
        self.log:  list[dict]  = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pct_used(self) -> float:
        if self.total <= 0:
            return 0.0
        return round(self.spent / self.total * 100, 1)

    def _warnings(self) -> list[str]:
        warnings = []
        pct = self._pct_used()
        if self.total <= 0:
            warnings.append(
                "No total budget set. Call budget_tracker with action='set_total' first."
            )
        elif pct >= _OVER_PCT:
            over = round(self.spent - self.total, 2)
            warnings.append(
                f"OVER BUDGET by ${over:.2f}! "
                "Review planned expenses or adjust the budget."
            )
        elif pct >= _WARN_PCT:
            remaining = round(self.total - self.spent, 2)
            warnings.append(
                f"Budget warning: {pct}% spent. "
                f"Only ${remaining:.2f} remaining."
            )
        return warnings

    def _by_category(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for entry in self.log:
            cat = entry.get("category") or "misc"
            totals[cat] = round(totals.get(cat, 0) + entry["amount"], 2)
        return totals

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.total = 0.0
        self.spent = 0.0
        self.log   = []

    def get_status(self) -> dict:
        return {
            "budget_total":    self.total,
            "total_spent":     round(self.spent, 2),
            "remaining":       round(self.total - self.spent, 2),
            "percent_used":    self._pct_used(),
            "by_category":     self._by_category(),
            "log":             list(self.log),
            "warnings":        self._warnings(),
        }

    # ------------------------------------------------------------------
    # Tool entry point
    # ------------------------------------------------------------------

    def run(
        self,
        action:      str,
        amount:      float = 0.0,
        category:    str   = "misc",
        description: str   = "",
    ) -> dict:
        action = action.lower().strip()
        amount = float(amount)

        if action == "set_total":
            self.total = amount
            return {
                "status":       "budget set",
                "budget_total": self.total,
                "remaining":    round(self.total - self.spent, 2),
                "warnings":     self._warnings(),
            }

        if action in ("add", "deduct"):
            self.spent += amount
            self.log.append({
                "category":    category or "misc",
                "amount":      amount,
                "description": description,
            })
            result = {
                "status":        "recorded",
                "category":      category,
                "amount_added":  amount,
                "total_spent":   round(self.spent, 2),
                "remaining":     round(self.total - self.spent, 2),
                "percent_used":  self._pct_used(),
                "budget_total":  self.total,
                "warnings":      self._warnings(),
            }
            return result

        if action == "status":
            return self.get_status()

        if action == "reset":
            self.spent = 0.0
            self.log   = []
            return {
                "status":       "spends cleared",
                "budget_total": self.total,
                "warnings":     self._warnings(),
            }

        return {
            "error": (
                f"Unknown action '{action}'. "
                "Use: set_total | add | deduct | status | reset"
            )
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

BUDGET = BudgetTracker()


def run(
    action:      str,
    amount:      float = 0.0,
    category:    str   = "misc",
    description: str   = "",
) -> dict:
    return BUDGET.run(action, amount, category, description)
