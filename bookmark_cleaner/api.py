from openai import OpenAI

from bookmark_cleaner import constant


def call_chat(inputs: list[dict[str, str]]) -> str:
    return [
        choice.message.content
        for choice in OpenAI()
        .chat.completions.create(
            model=constant.MODEL_NM,
            messages=inputs,  # type:ignore
        )
        .choices
        if choice.message.content
    ][0]


def call_embed(txt: str | list[str]) -> list[list[float]]:
    msg = OpenAI().embeddings.create(model=constant.EMBED_MODEL_NM, input=txt)
    return [d.embedding for d in msg.data]
