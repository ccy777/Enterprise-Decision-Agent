from __future__ import annotations

from decision_agent.data.ground_truth import load_operations_questions
from decision_agent.data.sql_guard import SQLGuard


def test_operations_question_set_is_versioned_complete_and_recomputable() -> None:
    questions = load_operations_questions()
    assert len(questions) == 8
    assert [question.question_id for question in questions] == [
        f"ops-v1-q{index:02d}" for index in range(1, 9)
    ]
    assert all(question.expected_result["columns"] for question in questions)
    assert all(question.expected_result["rows"] for question in questions)
    assert all("SELECT" in question.verification_sql.upper() for question in questions)


def test_all_formal_recomputation_rules_pass_the_same_sql_guard() -> None:
    guard = SQLGuard(max_rows=200)
    rejected = {
        question.question_id: guard.validate(question.verification_sql).rejection_code
        for question in load_operations_questions()
        if not guard.validate(question.verification_sql).allowed
    }
    assert rejected == {}
