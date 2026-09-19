"""Query normalization：模型输出检索词，失败时退回原句或已有短语。"""

import sys
from pathlib import Path
from unittest.mock import patch

if str(Path(__file__).parent.parent.parent / "src") not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from agents.query_normalization import normalize_search_query

_CASES = {
    "帮我找一些医学图像分割相关的论文": "medical image segmentation",
    "帮我找一些胎儿超声半监督分割方面的论文": "semi-supervised fetal ultrasound segmentation",
    "找一些大模型推理加速相关研究": "LLM inference acceleration",
}


def test_chinese_queries_become_short_english_keywords():
    def fake_model(query: str) -> str:
        return _CASES[query]

    with patch("agents.query_normalization._invoke_model", side_effect=fake_model):
        for source, expected in _CASES.items():
            assert normalize_search_query(source) == expected


def test_english_keywords_are_not_rewritten():
    with patch(
        "agents.query_normalization._invoke_model",
        side_effect=AssertionError("english keywords should not call the model"),
    ):
        assert (
            normalize_search_query("semi-supervised fetal ultrasound segmentation")
            == "semi-supervised fetal ultrasound segmentation"
        )
        assert normalize_search_query("LLM inference acceleration") == "LLM inference acceleration"


def test_normalization_failure_falls_back_without_raising():
    with patch(
        "agents.query_normalization._invoke_model",
        side_effect=RuntimeError("embedding endpoint unavailable"),
    ):
        assert (
            normalize_search_query("帮我找一些医学图像分割相关的论文")
            == "medical image segmentation"
        )
        original = "帮我找一些胎儿超声半监督分割方面的论文"
        assert normalize_search_query(original) == original


def test_unusable_model_text_falls_back_to_original_query():
    with patch(
        "agents.query_normalization._invoke_model",
        return_value="这是解释，不是检索词。\nI would search for several papers.",
    ):
        original = "找一些大模型推理加速相关研究"
        assert normalize_search_query(original) == original
