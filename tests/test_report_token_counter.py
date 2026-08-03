from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def test_public_builder_composes_nontruncating_local_counter():
    from build_community_reports import DEFAULT_TOKENIZER_DIR, compose_report_service
    from obsidian_wiki.infrastructure.production_token_counter import LocalReportTokenCounter

    counter = LocalReportTokenCounter(DEFAULT_TOKENIZER_DIR)
    assert counter.count("word " * 1000) == 1002
    assert "truncation=false" in counter.identity
    assert "special=true" in counter.identity
    # Composition is deliberately independent of WikiIndex/SentenceTransformer.
    service = compose_report_service(Path("."))
    assert service._token_counter.identity == counter.identity


def test_unavailable_counter_fails_closed(tmp_path):
    from obsidian_wiki.infrastructure.production_token_counter import LocalReportTokenCounter, TokenCounterUnavailable

    try:
        LocalReportTokenCounter(tmp_path / "missing")
    except TokenCounterUnavailable as exc:
        assert "token_counter_unavailable" in str(exc)
    else:
        raise AssertionError("missing local tokenizer must not use a character-count fallback")
