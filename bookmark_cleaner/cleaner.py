import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm  # type:ignore

from bookmark_cleaner import api, cluster, generate, summarize, url, utils

load_dotenv()


class OrganizedURL(BaseModel):
    url: str
    url_nm: str
    folder_nm: str


class URL(BaseModel):
    url: str
    url_nm: str
    summarized: str


class Cleaner:
    def __init__(
        self,
        urls: list[str],
        num_folders: int = 5,
    ) -> None:
        self.urls = urls
        self.num_folders = num_folders

    def cleanse(self, debug=True) -> list[OrganizedURL]:
        # read documents from urls
        data = utils.crawl_urls(self.urls)

        # summarize & naming it
        urls: list[URL] = []
        for idx, doc in (
            tqdm(enumerate(data), total=len(data)) if debug else enumerate(data)
        ):
            summarized = summarize.summarize_long_document(
                utils.split_text_with_token_length(
                    txt=doc,
                    token_length=4000,
                ),
                head=1,
            )
            url_nm = generate.generate_url_name(url.urls[idx], summarized)
            urls.append(
                URL(
                    url=url.urls[idx],
                    url_nm=url_nm,
                    summarized=summarized,
                )
            )

        # embedding & clustering
        embeds = np.array(
            api.call_embed(txt=[obj.summarized for obj in urls])
        ).squeeze()
        labels = cluster.cluster_embeds(embeds, self.num_folders)

        result: list[OrganizedURL] = []
        url_nms: list[str] = [obj.url_nm for obj in urls]

        label_folder_dict: dict[int, str] = {}
        for li in set(labels):
            folder_name = generate.generate_folder_name(
                [url_nms[idx] for idx, label in enumerate(labels) if label == li]
            )
            label_folder_dict[li] = folder_name

        # get result
        result += [
            OrganizedURL(
                url=obj.url,
                url_nm=obj.url_nm,
                folder_nm=label_folder_dict[label],
            )
            for obj, label in zip(urls, labels)
        ]

        return result


if __name__ == "__main__":
    result = Cleaner(urls=url.files[:]).cleanse()
    print(result[0].model_dump())
