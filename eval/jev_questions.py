"""The two questions the Jev-needle experiments ask, with the state fields inlined.

jev-decision-bench's state is a dict {documents, color, object, code}; ChunkLaya takes a string, so
color/object/code go into the question text and only `documents` is chunked.
"""

def noul_q(color, obj, code):
    """The benchmark's own question, with the state fields inlined."""
    return {
        "type": "noul",
        "instructions": (
            f"The text is a long collection of unrelated passages. Somewhere in it there is exactly "
            f"one sentence of the form \"The access code for the <color> <object> locker is "
            f"<four-digit code>.\" Does the text state that the access code for the {color} {obj} "
            f"locker is {code}?"
        ),
        "criteria": {
            "true": f"The text contains a sentence saying the access code for the {color} {obj} "
                    f"locker is exactly {code}.",
            "false": f"The text states a different access code for the {color} {obj} locker, or "
                     f"states no access code for it at all.",
        },
    }


def choice_q(color, obj, code):
    """Per-passage detector. Scoped to one excerpt, because that is all a chunk is."""
    return {
        "type": "choice",
        "instructions": (
            f"This passage is one excerpt from a longer document. Does this passage state that the "
            f"access code for the {color} {obj} locker is {code}?"
        ),
        "criteria": {
            "yes": f"This passage contains a sentence saying the access code for the {color} {obj} "
                   f"locker is exactly {code}.",
            "no": f"This passage does not mention the access code for the {color} {obj} locker, or "
                  f"gives a code other than {code}.",
        },
    }

