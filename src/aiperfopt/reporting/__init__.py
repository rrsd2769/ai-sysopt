"""
Reporting / Analytics.

Scoped strictly to reinforcing the project's three flagship pillars rather than functioning as generic dashboarding:
every summary this package produces ties back to prediction accuracy,
diagnosis/explainability, or recommendation traceability, not raw
metric dumping for its own sake. System-metrics context is included
only as background for interpreting the other three, matching the same
ordering the rest of the project uses (Prediction > Diagnosis/
Explainability > Recommendation).

Like every other subsystem, this one contains the actual computation;
``api/routers/reporting.py`` is a thin wrapper over it (the rule that
the API is the only access path is about frontends reaching into storage
directly, not about where business logic lives -- the API layer itself
still delegates to subsystem code, the same way ``retraining.py``'s
router delegates to ``aiperfopt.lifecycle``).
"""
