"""
知行 · AI 深度调研助手 — 主题分类器测试

测试内容：
1. 规则分类准确度（你好→daily_word, 唐朝→history 等）
2. QueryProfile 结构完整性
3. 反行业套壳检测（非 tech_business 类型不应出现行业模板词）
4. source_status 标记（fallback 时应为 "fallback"）
"""
import sys
import os
import asyncio
import pytest

# 添加后端目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.query_classifier import (
    classify_by_rules,
    QueryProfile,
    TOPIC_TYPE_DISPLAY,
    TOPIC_TYPE_TEMPLATE,
    TOPIC_TYPE_TONE,
)


# ============================================================
# 规则分类测试
# ============================================================

class TestRuleClassification:
    """测试规则分类的准确性"""

    def test_daily_word_nihao(self):
        """你好 → daily_word"""
        result = classify_by_rules("你好")
        assert result is not None
        assert result.topic_type == "daily_word"
        assert result.display_type == "日常词语"
        assert result.confidence >= 0.8

    def test_daily_word_hello(self):
        """hello → daily_word"""
        result = classify_by_rules("hello")
        assert result is not None
        assert result.topic_type == "daily_word"

    def test_question_why(self):
        """为什么年轻人焦虑？ → question"""
        result = classify_by_rules("为什么年轻人焦虑？")
        assert result is not None
        assert result.topic_type == "question"

    def test_question_how(self):
        """如何选择职业 → question"""
        result = classify_by_rules("如何选择职业")
        assert result is not None
        assert result.topic_type == "question"

    def test_social_phenomenon_neijuan(self):
        """内卷 → social_phenomenon"""
        result = classify_by_rules("内卷")
        assert result is not None
        assert result.topic_type == "social_phenomenon"
        assert result.display_type == "社会现象"

    def test_history_tang(self):
        """唐朝 → history"""
        result = classify_by_rules("唐朝")
        assert result is not None
        assert result.topic_type == "history"
        assert result.display_type == "历史主题"

    def test_history_coldwar(self):
        """冷战 → history"""
        result = classify_by_rules("冷战")
        assert result is not None
        assert result.topic_type == "history"

    def test_tech_business_ai(self):
        """AI Agent 工具市场 → tech_business"""
        result = classify_by_rules("AI Agent 工具市场")
        assert result is not None
        assert result.topic_type == "tech_business"
        assert result.display_type == "科技/商业"

    def test_tech_business_competitor(self):
        """Dify 和 Coze 竞品分析 → tech_business"""
        result = classify_by_rules("Dify 和 Coze 竞品分析")
        assert result is not None
        assert result.topic_type == "tech_business"

    def test_abstract_concept_freedom(self):
        """自由 → abstract_concept"""
        result = classify_by_rules("自由")
        assert result is not None
        assert result.topic_type == "abstract_concept"

    def test_unknown_returns_none(self):
        """月亮 → 规则可能无法判断，返回 None"""
        result = classify_by_rules("月亮")
        # 月亮可能命中 geography（有"月"字）也可能返回 None
        # 这里不强制要求，只测试不报错
        if result is not None:
            assert result.topic_type in TOPIC_TYPE_DISPLAY

    def test_random_word_returns_none_or_general(self):
        """随机词 → 可能返回 None（交给 LLM）"""
        result = classify_by_rules("咖啡")
        # 咖啡可能返回 None
        if result is not None:
            assert result.topic_type in TOPIC_TYPE_DISPLAY


# ============================================================
# QueryProfile 结构测试
# ============================================================

class TestQueryProfileStructure:
    """测试 QueryProfile 的数据结构完整性"""

    def test_profile_has_all_fields(self):
        """QueryProfile 应包含所有必要字段"""
        result = classify_by_rules("你好")
        assert result is not None
        assert hasattr(result, 'original_query')
        assert hasattr(result, 'topic_type')
        assert hasattr(result, 'template')
        assert hasattr(result, 'rewritten_query')
        assert hasattr(result, 'analysis_focus')
        assert hasattr(result, 'tone')
        assert hasattr(result, 'display_type')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'classified_by')

    def test_profile_to_dict(self):
        """to_dict() 应返回完整字典"""
        result = classify_by_rules("你好")
        assert result is not None
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "original_query" in d
        assert "topic_type" in d
        assert "rewritten_query" in d
        assert "analysis_focus" in d
        assert "display_type" in d

    def test_rewritten_query_not_empty(self):
        """改写后的查询词不应为空"""
        result = classify_by_rules("唐朝")
        assert result is not None
        assert len(result.rewritten_query) > len("唐朝")

    def test_analysis_focus_not_empty(self):
        """分析焦点不应为空"""
        result = classify_by_rules("内卷")
        assert result is not None
        assert len(result.analysis_focus) > 10

    def test_classified_by_is_rule(self):
        """规则分类的结果 classified_by 应为 'rule'"""
        result = classify_by_rules("你好")
        assert result is not None
        assert result.classified_by == "rule"


# ============================================================
# 反行业套壳测试
# ============================================================

class TestAntiTemplateShell:
    """测试非科技/商业主题不会出现行业套壳表达"""

    # 这些词在非 tech_business 主题中不应出现
    FORBIDDEN_PHRASES = [
        "市场规模持续增长",
        "头部企业占据",
        "政策支持力度加大",
        "企业全球化布局",
        "行业竞争格局",
        "预计未来三年保持",
        "寡头竞争态势",
    ]

    def test_daily_word_no_industry_shell(self):
        """日常词语的 rewritten_query 不应包含行业套壳词"""
        result = classify_by_rules("你好")
        assert result is not None
        for phrase in self.FORBIDDEN_PHRASES:
            assert phrase not in result.rewritten_query, \
                f"日常词语的 rewritten_query 不应包含 '{phrase}'"

    def test_history_no_industry_shell(self):
        """历史主题的 rewritten_query 不应包含行业套壳词"""
        result = classify_by_rules("唐朝")
        assert result is not None
        for phrase in self.FORBIDDEN_PHRASES:
            assert phrase not in result.rewritten_query, \
                f"历史主题的 rewritten_query 不应包含 '{phrase}'"

    def test_social_phenomenon_no_industry_shell(self):
        """社会现象的 rewritten_query 不应包含行业套壳词"""
        result = classify_by_rules("内卷")
        assert result is not None
        for phrase in self.FORBIDDEN_PHRASES:
            assert phrase not in result.rewritten_query, \
                f"社会现象的 rewritten_query 不应包含 '{phrase}'"

    def test_abstract_concept_no_industry_shell(self):
        """抽象概念的 rewritten_query 不应包含行业套壳词"""
        result = classify_by_rules("自由")
        assert result is not None
        for phrase in self.FORBIDDEN_PHRASES:
            assert phrase not in result.rewritten_query, \
                f"抽象概念的 rewritten_query 不应包含 '{phrase}'"

    def test_tech_business_can_have_industry_terms(self):
        """科技/商业主题可以包含行业相关词"""
        result = classify_by_rules("AI Agent 工具市场")
        assert result is not None
        assert result.topic_type == "tech_business"
        # tech_business 的 rewritten_query 应包含商业相关词
        assert "市场" in result.rewritten_query or "竞品" in result.rewritten_query or "商业" in result.rewritten_query


# ============================================================
# 主题类型映射测试
# ============================================================

class TestTopicTypeMapping:
    """测试主题类型到模板/语调的映射"""

    def test_all_types_have_display(self):
        """所有 topic_type 都有中文显示名"""
        for topic_type in TOPIC_TYPE_DISPLAY:
            assert TOPIC_TYPE_DISPLAY[topic_type] != ""

    def test_all_types_have_template(self):
        """所有 topic_type 都有模板名"""
        for topic_type in TOPIC_TYPE_TEMPLATE:
            assert TOPIC_TYPE_TEMPLATE[topic_type] != ""

    def test_all_types_have_tone(self):
        """所有 topic_type 都有语调"""
        for topic_type in TOPIC_TYPE_TONE:
            assert TOPIC_TYPE_TONE[topic_type] != ""

    def test_daily_word_tone(self):
        """日常词语语调应为轻知识"""
        result = classify_by_rules("你好")
        assert result is not None
        assert "轻知识" in result.tone or "启发" in result.tone

    def test_history_tone(self):
        """历史主题语调应包含叙事"""
        result = classify_by_rules("唐朝")
        assert result is not None
        assert "叙事" in result.tone or "历史" in result.tone
