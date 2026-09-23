import json

from answer_synthesis import build_answer_prompt, parse_answer_response, synthesize_answer

PASSAGES = [
    {"fileName": "Lock Desk Fee Schedule.pdf", "sectionTitle": "Extension Fees", "pageRange": "1",
     "content": "Policy A Agency per day -0.010"},
    {"fileName": "Chase VA Guide.pdf", "sectionTitle": "Qualifying Ratios", "pageRange": "3",
     "content": "Maximum 50% DTI"},
]


def test_prompt_numbers_each_passage_and_includes_the_question():
    prompt = build_answer_prompt("What is the lock extension fee?", PASSAGES)

    assert "What is the lock extension fee?" in prompt
    assert "[1] Lock Desk Fee Schedule.pdf | Extension Fees | pages 1" in prompt
    assert "Policy A Agency per day -0.010" in prompt
    assert "[2] Chase VA Guide.pdf | Qualifying Ratios | pages 3" in prompt


def test_parse_accepts_json_wrapped_in_a_code_fence():
    text = '```json\n{"found": true, "answer": "It is -0.010 per day [1].", "citations": [1]}\n```'

    assert parse_answer_response(text, passage_count=2) == {
        "found": True, "answer": "It is -0.010 per day [1].", "citations": [1],
    }


def test_parse_drops_citations_that_point_at_no_passage():
    text = json.dumps({"found": True, "answer": "x", "citations": [2, 7, 0, "1"]})

    assert parse_answer_response(text, passage_count=2)["citations"] == [2]


def test_parse_keeps_a_not_found_verdict():
    text = json.dumps({"found": False, "answer": "", "citations": []})

    assert parse_answer_response(text, passage_count=2) == {"found": False, "answer": "", "citations": []}


def test_parse_rejects_malformed_output():
    assert parse_answer_response("I think the answer is 50%", passage_count=2) is None
    assert parse_answer_response('{"answer": "no found flag"}', passage_count=2) is None
    assert parse_answer_response("", passage_count=2) is None


def test_synthesize_returns_none_so_callers_can_fall_back_when_the_model_fails():
    def failing_completion(prompt):
        raise RuntimeError("provider down")

    assert synthesize_answer("q", PASSAGES, failing_completion) is None
    assert synthesize_answer("q", [], lambda prompt: '{"found": true}') is None


def test_synthesize_parses_the_model_reply():
    reply = json.dumps({"found": True, "answer": "Chase allows up to 50% DTI [2].", "citations": [2]})

    assert synthesize_answer("max DTI?", PASSAGES, lambda prompt: reply) == {
        "found": True, "answer": "Chase allows up to 50% DTI [2].", "citations": [2],
    }
