"""
TOOL 4 — budget_tracker

Stateful ledger: supports add_expense, get_summary, reset, set_budget.
Returns full itemized breakdown, per-city totals, category totals,
remaining budget, recommendations, and exchange-rate context.
"""
from __future__ import annotations

from typing import Any


# In-process state: a single shared ledger (sufficient for one conversation).
_budget_state: dict[str, Any] = {
    "total_budget_usd": 0.0,
    "currency": "USD",
    "travellers": 1,
    "trip_name": "Unnamed trip",
    "expenses": [],
    "notes": [],
}


def budget_tracker(
    action: str,
    amount: float = 0.0,
    category: str = "other",
    city: str = "",
    description: str = "",
    total_budget: float = 0.0,
    travellers: int = 1,
    trip_name: str = "",
    currency: str = "USD",
) -> dict:
    """
    Stateful budget tracker for a multi-city trip.

    Actions
    -------
    set_budget   : initialise or update total_budget, travellers, trip_name
    add_expense  : add a line item; amount in USD
    get_summary  : return full ledger + per-city and per-category totals
    reset        : clear all expenses (keeps budget setting)
    """
    global _budget_state

    if action == "set_budget":
        _budget_state["total_budget_usd"] = float(total_budget) if total_budget else _budget_state["total_budget_usd"]
        _budget_state["travellers"] = max(1, travellers)
        _budget_state["currency"] = currency.upper()
        if trip_name:
            _budget_state["trip_name"] = trip_name
        return {
            "action": "set_budget",
            "status": "ok",
            "budget_set_usd": _budget_state["total_budget_usd"],
            "travellers": _budget_state["travellers"],
            "trip_name": _budget_state["trip_name"],
            "per_person_usd": round(_budget_state["total_budget_usd"] / _budget_state["travellers"], 2),
        }

    if action == "add_expense":
        expense = {
            "id": len(_budget_state["expenses"]) + 1,
            "amount_usd": round(float(amount), 2),
            "category": category.lower().strip(),
            "city": city.strip() or "unspecified",
            "description": description or f"{category} expense",
        }
        _budget_state["expenses"].append(expense)
        total_spent = sum(e["amount_usd"] for e in _budget_state["expenses"])
        remaining = _budget_state["total_budget_usd"] - total_spent
        budget = _budget_state["total_budget_usd"]

        if budget > 0:
            spent_pct = (total_spent / budget) * 100
            if spent_pct >= 100:
                status = "OVER_BUDGET"
                warning = f"ALERT: You are ${abs(remaining):.2f} OVER budget!"
            elif spent_pct >= 85:
                status = "critical"
                warning = f"WARNING: {spent_pct:.1f}% of budget used. Only ${remaining:.2f} remaining."
            elif spent_pct >= 70:
                status = "caution"
                warning = f"NOTE: {spent_pct:.1f}% of budget used. ${remaining:.2f} remaining."
            else:
                status = "healthy"
                warning = None
        else:
            status = "no_budget_set"
            warning = "No total budget set — call set_budget first."

        result = {
            "action": "add_expense",
            "expense_added": expense,
            "running_total_spent_usd": round(total_spent, 2),
            "total_budget_usd": budget,
            "remaining_usd": round(remaining, 2),
            "budget_status": status,
            "total_expenses_count": len(_budget_state["expenses"]),
        }
        if warning:
            result["warning"] = warning
        return result

    if action == "get_summary":
        expenses = _budget_state["expenses"]
        total_spent = sum(e["amount_usd"] for e in expenses)
        budget = _budget_state["total_budget_usd"]
        remaining = budget - total_spent
        travellers = _budget_state["travellers"]

        categories: dict[str, float] = {}
        for e in expenses:
            categories[e["category"]] = categories.get(e["category"], 0.0) + e["amount_usd"]

        cities: dict[str, dict] = {}
        for e in expenses:
            c = e["city"]
            if c not in cities:
                cities[c] = {"total_usd": 0.0, "expense_count": 0, "categories": {}}
            cities[c]["total_usd"] = round(cities[c]["total_usd"] + e["amount_usd"], 2)
            cities[c]["expense_count"] += 1
            cities[c]["categories"][e["category"]] = round(
                cities[c]["categories"].get(e["category"], 0.0) + e["amount_usd"], 2
            )

        if budget > 0:
            spent_pct = (total_spent / budget) * 100
            if spent_pct >= 100:
                health = "OVER_BUDGET"
                health_note = f"Over budget by ${abs(remaining):.2f}"
            elif spent_pct >= 85:
                health = "critical"
                health_note = f"Only ${remaining:.2f} left ({100-spent_pct:.1f}% of budget)"
            elif spent_pct >= 70:
                health = "caution"
                health_note = f"${remaining:.2f} remaining — budget tight"
            elif spent_pct >= 50:
                health = "moderate"
                health_note = f"${remaining:.2f} remaining — on track"
            else:
                health = "healthy"
                health_note = f"${remaining:.2f} remaining — plenty of budget left"
        else:
            health = "no_budget_set"
            health_note = "Set a budget with action='set_budget'"

        recommendations = []
        if budget > 0 and remaining > 0:
            daily_remaining_estimate = remaining / max(1, 3)
            recommendations.append(f"Estimated daily remaining budget: ~${daily_remaining_estimate:.0f}/day (assuming 3 days left)")
            if remaining < 200:
                recommendations.append("Consider switching to budget accommodation (hostels, guesthouses) for remaining stay.")
                recommendations.append("Use local public transport instead of taxis.")
                recommendations.append("Eat at local markets and street food stalls (50-70% cheaper than restaurants).")
            elif remaining < 500:
                recommendations.append("Choose 3-star hotels rather than 4-star for remaining cities.")
                recommendations.append("Limit fine dining to 1 special meal; eat local otherwise.")
            else:
                recommendations.append("Budget is in good shape — room for upgrades if desired.")

        exchange_rates = {
            "EUR": {"rate": 0.93, "note": "Paris, Amsterdam, Berlin"},
            "JPY": {"rate": 154.0, "note": "Tokyo, Kyoto"},
            "IDR": {"rate": 16200, "note": "Bali"},
            "INR": {"rate": 83.5, "note": "India (Delhi, Dehradun, Manali)"},
            "CHF": {"rate": 0.90, "note": "Switzerland (Zurich, Geneva)"},
        }

        return {
            "action": "get_summary",
            "trip_name": _budget_state["trip_name"],
            "travellers": travellers,
            "total_budget_usd": budget,
            "total_spent_usd": round(total_spent, 2),
            "remaining_usd": round(remaining, 2),
            "per_person_spent_usd": round(total_spent / travellers, 2),
            "per_person_remaining_usd": round(remaining / travellers, 2),
            "budget_health": health,
            "budget_health_note": health_note,
            "spent_percentage": round((total_spent / budget * 100), 1) if budget > 0 else None,
            "by_category": {k: round(v, 2) for k, v in sorted(categories.items(), key=lambda x: -x[1])},
            "by_city": cities,
            "all_expenses": expenses,
            "recommendations": recommendations,
            "exchange_rates_apr2026": exchange_rates,
            "metadata": {
                "source": "mock_budget_tracker_v1",
                "note": "All amounts in USD. Exchange rates approximate as of April 2026.",
            },
        }

    if action == "reset":
        _budget_state["expenses"] = []
        _budget_state["notes"] = []
        return {
            "action": "reset",
            "status": "ok",
            "message": "All expenses cleared. Budget settings retained.",
            "total_budget_usd": _budget_state["total_budget_usd"],
        }

    return {"error": f"Unknown action '{action}'. Use: set_budget, add_expense, get_summary, reset"}


SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "budget_tracker",
        "description": (
            "Track trip budget across cities and categories. "
            "Supports: set_budget (initialise), add_expense (log a cost), "
            "get_summary (full breakdown), reset (clear expenses)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["set_budget", "add_expense", "get_summary", "reset"],
                    "description": "What to do",
                },
                "amount": {
                    "type": "number",
                    "description": "Expense amount in USD (for add_expense)",
                },
                "category": {
                    "type": "string",
                    "enum": ["flights", "hotels", "food", "activities", "transport", "visa", "insurance", "shopping", "other"],
                    "description": "Expense category (for add_expense)",
                },
                "city": {
                    "type": "string",
                    "description": "City where expense was incurred (for add_expense)",
                },
                "description": {
                    "type": "string",
                    "description": "Free-text description of the expense",
                },
                "total_budget": {
                    "type": "number",
                    "description": "Total trip budget in USD (for set_budget)",
                },
                "travellers": {
                    "type": "integer",
                    "description": "Number of travellers (for set_budget)",
                },
                "trip_name": {
                    "type": "string",
                    "description": "Name for this trip (for set_budget)",
                },
            },
            "required": ["action"],
        },
    },
}
