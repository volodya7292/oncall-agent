from oncall.telegram_format import split_messenger_reply


def test_split_messenger_reply_uses_blank_lines_as_chat_bubble_boundaries():
    """The operator can intentionally emit several short Telegram bubbles."""
    assert split_messenger_reply("аха\n\nвот это да\n\nа ты как?") == [
        "аха", "вот это да", "а ты как?",
    ]


def test_split_messenger_reply_keeps_single_newlines_in_one_bubble():
    assert split_messenger_reply("one\ntwo\nthree") == ["one\ntwo\nthree"]


def test_split_messenger_reply_never_cuts_a_thought_by_length():
    """Regression: a 90-char target once cut replies mid-sentence ("..., а" /
    "не з ..."), producing bubbles no person would send."""
    sentence = "вода выглядит чистой, но ручей течёт низко через мох и густые заросли, а не с ледника"
    assert split_messenger_reply(sentence * 3) == [sentence * 3]
