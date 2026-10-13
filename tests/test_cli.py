import json

from email_classifier import cli
from email_classifier.cli import EXIT_CONFIG, EXIT_OK, main


def test_providers_command(capsys, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "AWS_BEARER_TOKEN_BEDROCK", "ANTHROPIC_AWS_API_KEY", "CLAUDE_PLATFORM"):
        monkeypatch.delenv(var, raising=False)
    assert main(["--env-file", "", "providers"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "jev" in out and "NOT CONFIGURED" in out and "sonnet" in out


def test_validate_dataset_command(capsys):
    assert main(["validate-dataset"]) == EXIT_OK
    assert "Dataset OK" in capsys.readouterr().out


def test_evaluate_with_unconfigured_jev_exits_with_config_error(capsys, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert main(["--env-file", "", "evaluate", "--provider", "jev", "--limit", "1"]) == EXIT_CONFIG
    assert "not configured" in capsys.readouterr().err


def test_classify_with_injected_provider(capsys, monkeypatch, make_provider):
    from conftest import make_response

    provider, _ = make_provider([make_response()])
    monkeypatch.setattr(cli, "get_provider", lambda name, **kw: provider)
    code = main(["--env-file", "", "classify", "--no-cache", "--subject", "Charged twice",
                 "--body", "Please refund the duplicate payment today."])
    assert code == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["classification"]["email_id"] == "ADHOC-001"
    assert payload["classification"]["department"] == "billing_payments"
    assert payload["meta"]["model"] == "claude-sonnet-5-5"
