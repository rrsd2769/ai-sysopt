"""
Reporting / Analytics (Section 7.1 item 14, FR-14.1/FR-14.2, ADL-22).

Scoped strictly to reinforcing the project's three flagship pillars
(Section 6) rather than functioning as generic dashboarding (ADL-22):
every summary this package produces ties back to prediction accuracy,
diagnosis/explainability, or recommendation traceability, not raw
metric dumping for its own sake. System-metrics context is included
only as background for interpreting the other three, matching the same
ordering the rest of the project uses (Prediction > Diagnosis/
Explainability > Recommendation, Section 6).

Like every other subsystem, this one contains the actual computation;
``api/routers/reporting.py`` is a thin wrapper over it (FR-12.4's "API
is the only access path" is about frontends reaching into storage
directly, not about where business logic lives -- the API layer itself
still delegates to subsystem code, the same way ``retraining.py``'s
router delegates to ``aiperfopt.lifecycle``).
"""
