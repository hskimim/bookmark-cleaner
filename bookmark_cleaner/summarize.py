from bookmark_cleaner import api


def _summarize_with_chatmodel(txt: str) -> str:
    return api.call_chat(
        [
            {"role": "system", "content": "summarize given document." ""},
            {"role": "user", "content": txt},
        ]
    )


def _reduce_documents_with_summarization(txts: list[str]) -> str:
    return api.call_chat(
        [
            {
                "role": "system",
                "content": f"""there are {len(txts)} number of documents. These are summaries of each part of the same document. 
                 Combine these contents and summarize them so that the result is similar to summarizing the entire document. 
                 """,
            },
            {"role": "user", "content": str(txts)},
        ]
    )


def summarize_long_document(txts: list[str], head: int | None = None) -> str:
    summarized_subdocs = [_summarize_with_chatmodel(txt) for txt in txts[:head]]
    return (
        _reduce_documents_with_summarization(summarized_subdocs)
        if head != 1
        else summarized_subdocs[0]
    )
