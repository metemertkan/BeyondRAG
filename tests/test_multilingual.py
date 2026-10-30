"""Unit tests for language detection, query translation, and language-independent retrieval."""

import pytest
from src.rag.cross_lingual import (
    LLMQueryTranslator,
    MultilingualQueryAdapter,
    RuleBasedLanguageDetector,
)


def test_language_detector_multilingual():
    detector = RuleBasedLanguageDetector()

    assert detector.detect("What is the minimum cash reserve?") == "en"
    assert detector.detect("香港天文台八號風球警告") == "zh"
    assert detector.detect("台風第8号の警報が出ました") == "ja"
    assert detector.detect("태풍 경보 발령") == "ko"
    assert detector.detect("¿Cuál es la política de asistencia híbrida?") == "es"
    assert detector.detect("Quelle est la politique de télétravail?") == "fr"
    assert detector.detect("Wie lautet die Richtlinie für Notfälle?") == "de"
    assert detector.detect("Какова минимальная резервная сумма?") == "ru"


def test_accented_latin_characters_handling():
    detector = RuleBasedLanguageDetector()
    # Resume / café / naïve should not be misclassified as Asian or unsupported
    assert detector.detect("Please submit your résumé and CV for the engineering role.") == "en"


def test_query_translator_noop_when_same_language():
    translator = LLMQueryTranslator(enabled=True)
    res = translator.translate("Core working hours", "en", "en")
    assert res == "Core working hours"


def test_query_translator_disabled_returns_original():
    translator = LLMQueryTranslator(enabled=False)
    res = translator.translate("八號風球", "zh", "en")
    assert res == "八號風球"


def test_multilingual_adapter_preserves_query_when_fallback_disabled():
    adapter = MultilingualQueryAdapter()
    query, concepts = adapter.prepare_query("八號風球", allow_llm_fallback=False)
    assert query == "八號風球"
