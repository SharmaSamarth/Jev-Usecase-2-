"""Fixed classification taxonomy shared by every provider.

This module is the single source of truth for allowed labels and their
definitions. Schemas, prompts, the JSON schema sent to models, and dataset
validation all derive from it, so labels are never redefined elsewhere.
"""

from __future__ import annotations

from enum import Enum


class _Label(str, Enum):
    """String-valued enum whose str() is the raw label."""

    def __str__(self) -> str:
        return self.value


class Tone(_Label):
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"
    ANXIOUS = "anxious"
    CONFUSED = "confused"
    APPRECIATIVE = "appreciative"
    ASSERTIVE = "assertive"
    MIXED = "mixed"
    UNCLEAR = "unclear"


class Urgency(_Label):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Department(_Label):
    BILLING_PAYMENTS = "billing_payments"
    TECHNICAL_SUPPORT = "technical_support"
    ACCOUNT_ACCESS = "account_access"
    SALES = "sales"
    CUSTOMER_SUCCESS = "customer_success"
    SECURITY_FRAUD = "security_fraud"
    LEGAL_COMPLIANCE = "legal_compliance"
    HR_PEOPLE = "hr_people"
    OPERATIONS = "operations"
    GENERAL_OTHER = "general_other"


class Intent(_Label):
    QUESTION = "question"
    SERVICE_REQUEST = "service_request"
    COMPLAINT = "complaint"
    INCIDENT_REPORT = "incident_report"
    REFUND_REQUEST = "refund_request"
    CANCELLATION = "cancellation"
    FEEDBACK = "feedback"
    SALES_LEAD = "sales_lead"
    INFORMATION_ONLY = "information_only"
    OTHER = "other"


class Action(_Label):
    IMMEDIATE_ESCALATION = "immediate_escalation"
    RESPOND_TODAY = "respond_today"
    STANDARD_QUEUE = "standard_queue"
    NO_ACTION = "no_action"
    HUMAN_REVIEW = "human_review"


# Ordered from most to least severe. Used to detect under-triage.
URGENCY_SEVERITY: dict[Urgency, int] = {
    Urgency.CRITICAL: 3,
    Urgency.HIGH: 2,
    Urgency.MEDIUM: 1,
    Urgency.LOW: 0,
}

# Categorical fields evaluated by the framework, mapped to their label enum.
CATEGORICAL_FIELDS: dict[str, type[_Label]] = {
    "tone": Tone,
    "urgency": Urgency,
    "department": Department,
    "intent": Intent,
    "recommended_action": Action,
}


DEFINITIONS: dict[str, dict[str, str]] = {
    "tone": {
        Tone.NEUTRAL: "Matter-of-fact, no notable emotion.",
        Tone.POSITIVE: "Upbeat or pleased, without specifically thanking anyone.",
        Tone.FRUSTRATED: "Annoyed or exasperated, typically about a repeated or unresolved problem, but still controlled.",
        Tone.ANGRY: "Hostile, accusatory, or furious language (insults, threats, heavy capitals or exclamation).",
        Tone.ANXIOUS: "Worried, stressed, or fearful about consequences.",
        Tone.CONFUSED: "Unsure what is happening or what to do; asks for clarification.",
        Tone.APPRECIATIVE: "Expresses thanks or praise toward the company or staff.",
        Tone.ASSERTIVE: "Firm, direct, demanding specific outcomes or deadlines without hostility.",
        Tone.MIXED: "Clearly combines two or more distinct tones (for example praise followed by anger).",
        Tone.UNCLEAR: "Too short or garbled to judge the tone.",
    },
    "urgency": {
        Urgency.CRITICAL: "An active or imminent security incident, fraud event, major outage, or another issue requiring immediate escalation.",
        Urgency.HIGH: "A significant issue requiring prompt attention because of substantial impact, an approaching deadline, or serious disruption.",
        Urgency.MEDIUM: "A normal service issue or request that should be addressed within the standard working queue.",
        Urgency.LOW: "Routine inquiries, general information, non-urgent feedback, or requests with no significant time pressure.",
    },
    "department": {
        Department.BILLING_PAYMENTS: "Charges, invoices, refunds, payment methods, billing errors (cancellation requests themselves go to customer_success).",
        Department.TECHNICAL_SUPPORT: "Product bugs, application errors, integrations, performance problems, service outages.",
        Department.ACCOUNT_ACCESS: "Login failures, password resets, MFA problems, locked accounts, account ownership changes (without signs of compromise).",
        Department.SALES: "Pricing, quotes, purchasing, plan upgrades for prospective or expanding customers.",
        Department.CUSTOMER_SUCCESS: "Onboarding, adoption, renewals, subscription cancellations and retention, feedback and praise about the product or service experience.",
        Department.SECURITY_FRAUD: "Suspected account compromise, unauthorized access or transactions, fraud, phishing, data breaches, vulnerabilities.",
        Department.LEGAL_COMPLIANCE: "Contracts, legal notices, regulatory or privacy requests (e.g. GDPR/CCPA), subpoenas.",
        Department.HR_PEOPLE: "Employment, job applications, payroll for staff, workplace and employee matters.",
        Department.OPERATIONS: "Shipping, deliveries, logistics, vendors, facilities, physical fulfilment.",
        Department.GENERAL_OTHER: "Anything that does not fit another department or lacks enough information to route reliably.",
    },
    "intent": {
        Intent.QUESTION: "Asks for information or an explanation.",
        Intent.SERVICE_REQUEST: "Asks the company to perform a specific task or change (other than refund or cancellation).",
        Intent.COMPLAINT: "Expresses dissatisfaction and wants it acknowledged or fixed, without primarily reporting a new incident.",
        Intent.INCIDENT_REPORT: "Reports something that is broken, down, or a security/fraud event that is happening or has happened.",
        Intent.REFUND_REQUEST: "Asks for money back.",
        Intent.CANCELLATION: "Asks to cancel or terminate a subscription, order, or contract.",
        Intent.FEEDBACK: "Offers opinions, praise, or suggestions without a specific request.",
        Intent.SALES_LEAD: "A prospective or existing customer expresses buying interest.",
        Intent.INFORMATION_ONLY: "Shares information that requires no reply (FYI, confirmation, notice).",
        Intent.OTHER: "None of the above, or purpose cannot be determined.",
    },
    "recommended_action": {
        Action.IMMEDIATE_ESCALATION: "Page the owning team now. Use for all critical-urgency emails.",
        Action.RESPOND_TODAY: "Must be handled within the same business day. Use for high-urgency emails.",
        Action.STANDARD_QUEUE: "Normal working queue. Use for medium-urgency emails and for low-urgency emails that still need a reply.",
        Action.NO_ACTION: "Nothing needs to be done beyond filing (e.g. FYI, thank-you notes, confirmations).",
        Action.HUMAN_REVIEW: "Route to a human triager before any automated routing. Use when human review is required and urgency is not critical.",
    },
}


def allowed_values(field: str) -> list[str]:
    """Return the allowed string labels for a categorical field."""
    return [member.value for member in CATEGORICAL_FIELDS[field]]


def expected_action(urgency: Urgency, human_review_required: bool, needs_reply: bool) -> Action:
    """The action rule used for ground-truth labeling and stated in the prompt.

    Critical always escalates (a human is paged anyway). Otherwise an email
    flagged for review goes to a human before routing. The remaining mapping
    follows urgency, with low-urgency emails that need no reply getting
    ``no_action``.
    """
    if urgency is Urgency.CRITICAL:
        return Action.IMMEDIATE_ESCALATION
    if human_review_required:
        return Action.HUMAN_REVIEW
    if urgency is Urgency.HIGH:
        return Action.RESPOND_TODAY
    if urgency is Urgency.MEDIUM:
        return Action.STANDARD_QUEUE
    return Action.STANDARD_QUEUE if needs_reply else Action.NO_ACTION
