import pytest

from email_classifier.taxonomy import (
    CATEGORICAL_FIELDS,
    DEFINITIONS,
    Action,
    Department,
    Intent,
    Tone,
    Urgency,
    allowed_values,
    expected_action,
)

SPEC = {
    "tone": ["neutral", "positive", "frustrated", "angry", "anxious", "confused", "appreciative", "assertive", "mixed", "unclear"],
    "urgency": ["critical", "high", "medium", "low"],
    "department": ["billing_payments", "technical_support", "account_access", "sales", "customer_success",
                   "security_fraud", "legal_compliance", "hr_people", "operations", "general_other"],
    "intent": ["question", "service_request", "complaint", "incident_report", "refund_request", "cancellation",
               "feedback", "sales_lead", "information_only", "other"],
    "recommended_action": ["immediate_escalation", "respond_today", "standard_queue", "no_action", "human_review"],
}


@pytest.mark.parametrize("field", list(SPEC))
def test_labels_match_specification_exactly(field):
    assert allowed_values(field) == SPEC[field]


@pytest.mark.parametrize("field", list(CATEGORICAL_FIELDS))
def test_every_label_has_a_definition(field):
    assert set(DEFINITIONS[field]) == set(CATEGORICAL_FIELDS[field])
    assert all(text.strip() for text in DEFINITIONS[field].values())


def test_labels_render_as_raw_strings():
    assert str(Department.SECURITY_FRAUD) == "security_fraud"
    assert f"{Urgency.CRITICAL}" == "critical"


@pytest.mark.parametrize(
    "urgency, review, needs_reply, expected",
    [
        (Urgency.CRITICAL, False, True, Action.IMMEDIATE_ESCALATION),
        (Urgency.CRITICAL, True, True, Action.IMMEDIATE_ESCALATION),
        (Urgency.HIGH, True, True, Action.HUMAN_REVIEW),
        (Urgency.LOW, True, False, Action.HUMAN_REVIEW),
        (Urgency.HIGH, False, True, Action.RESPOND_TODAY),
        (Urgency.MEDIUM, False, True, Action.STANDARD_QUEUE),
        (Urgency.LOW, False, True, Action.STANDARD_QUEUE),
        (Urgency.LOW, False, False, Action.NO_ACTION),
    ],
)
def test_expected_action_rule(urgency, review, needs_reply, expected):
    assert expected_action(urgency, review, needs_reply) is expected


def test_enums_reject_unknown_labels():
    with pytest.raises(ValueError):
        Tone("furious")
    with pytest.raises(ValueError):
        Intent("refund")
