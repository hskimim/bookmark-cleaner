import tiktoken
from langchain_community.document_loaders import WebBaseLoader
from tqdm import tqdm

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


def crawl_urls(data: list[str]) -> list[str]:
    urls = [d for d in data if not d.endswith(".pdf")]
    fnames = [d for d in data if d.endswith(".pdf")]

    return [
        doc.page_content.replace("\n", "")
        for doc in WebBaseLoader(urls, continue_on_failure=True).load()
    ] + [crawl_pdf(file) for file in tqdm(fnames)]


def crawl_pdf(fname: str) -> str:
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(fname)
    docs = loader.load()
    result = ""
    for doc in docs:
        result += doc.page_content.replace("\n", "")
    return result
    return result
    return result
