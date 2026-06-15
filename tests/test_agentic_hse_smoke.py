from agentic_hse.approval import requires_approval


def test_requires_approval_threshold():
    assert requires_approval(10) is True
    assert requires_approval(9) is False
