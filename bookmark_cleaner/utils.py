import tiktoken
from langchain_community.document_loaders import WebBaseLoader

from bookmark_cleaner import constant

enc = tiktoken.encoding_for_model(constant.MODEL_NM)


def split_list(input_list, chunk_size):
    result_list = [
        input_list[i : i + chunk_size] for i in range(0, len(input_list), chunk_size)
    ]
    return result_list


def split_text_with_token_length(txt: str, token_length: int = 8000) -> list[str]:
    tokens: list[int] = enc.encode(txt)
    nested_ls = split_list(tokens, token_length)
    return [enc.decode(ls) for ls in nested_ls]


def get_token_length(txt: str) -> int:
    return len(enc.encode(txt))


def crawl_urls(urls: list[str]) -> list[str]:
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return [doc.page_content.replace("\n", "") for doc in docs]
