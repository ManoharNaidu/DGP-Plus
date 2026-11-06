import numpy as np

from dgp_plus.pipeline import WordTokenizer
from dgp_plus.prompt import MetapathBlock, build_prompt, format_block, load_template, truncate_to_tokens


def test_final_prompt_layout():
    blocks = [MetapathBlock("RTR-RSR", 2, " short summary ", np.array([1.666])), MetapathBlock("RSR", 0)]
    prompt = build_prompt(" a review ", blocks, ["rating"], "Is this fraud?", load_template("final_prompt"))
    assert prompt == ("Target: a review\nMetapaths:\n- RTR-RSR: short summary (neighbor mean: rating=1.67)\n"
                      "- RSR: no neighbors\nQuestion: Is this fraud?")


def test_no_numeric_part_when_summary_is_off():
    assert format_block(MetapathBlock("URU", 1, "text"), []) == "- URU: text"


def test_truncation_keeps_short_text_exact():
    tokenizer = WordTokenizer()
    assert truncate_to_tokens(tokenizer, "a  b", 5) == "a  b"
    assert truncate_to_tokens(tokenizer, "a b c", 2) == "a b"
