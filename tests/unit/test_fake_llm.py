from research_copilot.llm.fake import FakeLLM

from research_copilot.llm.client import Message


def test_fake_returns_replies_in_order_then_repeats_last():
    fake = FakeLLM(["one", "two"])
    messages = [Message("user", "hi")]

    assert fake.chat(messages).text == "one"
    assert fake.chat(messages).text == "two"
    assert fake.chat(messages).text == "two"


def test_fake_records_calls():
    fake = FakeLLM()
    fake.chat([Message("system", "be brief"), Message("user", "hello there")])

    assert len(fake.calls) == 1
    assert fake.calls[0][0].role == "system"
    assert fake.calls[0][1].content == "hello there"
