import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from sklearn.cluster import KMeans
from tqdm import tqdm

from bookmark_cleaner import constant, url

load_dotenv()

import requests
import tiktoken
from bs4 import BeautifulSoup
from openai import OpenAI


def http_call(url: str) -> str:  # make it with asnyc
    return BeautifulSoup(requests.get(url).text, "html.parser").text.replace("\n", "")


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


def get_token_length(txt: str) -> len:
    return len(enc.encode(txt))


def _summarize_with_chatmodel(txt: str) -> str:
    return [
        choice.message.content
        for choice in OpenAI()
        .chat.completions.create(
            model=constant.MODEL_NM,
            messages=[
                {"role": "system", "content": "summarize given document." ""},
                {"role": "user", "content": txt},
            ],
        )
        .choices
    ][0]


def _reduce_documents_with_summarization(txts: list[str]) -> str:
    return [
        choice.message.content
        for choice in OpenAI()
        .chat.completions.create(
            model=constant.MODEL_NM,
            messages=[
                {
                    "role": "system",
                    "content": f"""there are {len(txts)} number of documents. These are summaries of each part of the same document. 
                 Combine these contents and summarize them so that the result is similar to summarizing the entire document. 
                 """,
                },
                {"role": "user", "content": str(txts)},
            ],
        )
        .choices
    ][0]


def summarize_long_document(txts: list[str], head: int | None = None) -> str:
    return _reduce_documents_with_summarization(
        [_summarize_with_chatmodel(txt) for txt in txts[:head]]
    )


def generate_url_name(url: str, txt: str) -> str:
    msg = OpenAI().chat.completions.create(
        model=constant.MODEL_NM,
        messages=[
            {
                "role": "system",
                "content": f"""I will give you a summary of the document, so please create a title that sufficiently contains the information in this document.
          For reference, the url of this document is {url}. Use that information if it helps you come up with a title.""",
            },
            {"role": "user", "content": txt},
        ],
    )

    return [choice.message.content for choice in msg.choices][0]


def generate_folder_name(txts: list[str]) -> str:
    msg = OpenAI().chat.completions.create(
        model=constant.MODEL_NM,
        messages=[
            {
                "role": "system",
                "content": """"You must name a folder that will contain documents with the titles user going to give you.
         The most important thing is that you must be able to understand the characteristics of the files in the folder just by looking at it, and you must be able to distinguish them from files in other folders. 
         Express it compactly in below three words.""",
            },
            {"role": "user", "content": str(txts)},
        ],
    )
    return [choice.message.content for choice in msg.choices][0]


def openai_embed_call(txt: str | list[str]) -> list[list[float]]:
    msg = OpenAI().embeddings.create(model=constant.EMBED_MODEL_NM, input=txt)

    return [d.embedding for d in msg.data]


if __name__ == "__main__":
    # prepare data
    loader = WebBaseLoader(url.urls[:10])
    docs = loader.load()
    data: list[str] = [doc.page_content.replace("\n", "") for doc in docs]

    # summarize & naming it
    results = dict()

    for idx, doc in tqdm(enumerate(data), total=len(data)):
        result = summarize_long_document(
            split_text_with_token_length(
                txt=doc,
                token_length=8000,
            ),
            head=1,
        )
        name = generate_url_name(url.urls[idx], result)

        results[name] = result
        print(name)

    embeds = np.array(
        [openai_embed_call(txt=result) for result in results.values()]
    ).squeeze()
    cluster = KMeans(n_clusters=embeds.shape[0] // 3, n_init="auto").fit(embeds)
    labels = cluster.predict(embeds)
    folder_name = generate_folder_name(
        [list(results.keys())[idx] for idx, label in enumerate(labels) if label == 0]
    )
