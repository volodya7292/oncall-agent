from oncall.telegram_format import split_messenger_reply


def test_split_messenger_reply_uses_blank_lines_as_chat_bubble_boundaries():
    """The operator can intentionally emit several short Telegram bubbles."""
    assert split_messenger_reply("аха\n\nвот это да\n\nа ты как?") == [
        "аха", "вот это да", "а ты как?",
    ]


def test_split_messenger_reply_keeps_single_newlines_in_one_bubble():
    assert split_messenger_reply("one\ntwo\nthree") == ["one\ntwo\nthree"]


def test_split_messenger_reply_breaks_long_prose_at_a_word_boundary():
    text = "eins " * 20
    pieces = split_messenger_reply(text)

    assert len(pieces) == 2
    assert "".join(piece + " " for piece in pieces).split() == text.split()
    assert all(len(piece) <= 45 for piece in pieces)
