"""Frozen scoring policy for missing and invalid predictions.

This policy is fixed before any model comparison is interpreted.

strict
    Every ground-truth diagram is included. A missing file, invalid JSON,
    refusal, or truncated output is scored as an empty prediction. Supported
    categories therefore contribute F1 0. This is the coverage-corrected score.

conditional
    The mean is taken only over archived files that parse as JSON directly.
    This matches the historical compare/bleu/ged behavior, which skips missing
    files. It is survivorship-biased when the missing diagrams are the harder
    ones, so it is never the only number reported.

lenient
    Same denominator rule as conditional, except a JSON object wrapped in a
    code fence or surrounding prose is accepted. Outputs that still do not
    parse stay out of this mean and are counted in the audit.

Both strict and conditional are always emitted. The choice is not revised
after seeing which policy favors a particular model.
"""

POLICY_STRICT = "strict"
POLICY_CONDITIONAL = "conditional"
POLICY_LENIENT = "lenient"

# Direct JSON is required for conditional scoring. Lenient scoring additionally
# accepts one extracted object. Empty but valid schemas are real predictions.
CONDITIONAL_STATUSES = {"valid_json", "empty_schema"}
LENIENT_STATUSES = CONDITIONAL_STATUSES | {"lenient_extracted"}
