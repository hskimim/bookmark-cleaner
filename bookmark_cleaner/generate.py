from bookmark_cleaner import api


def generate_url_name(url: str, txt: str) -> str:
    return api.call_chat(
        [
            {
                "role": "system",
                "content": f"""I will give you a summary of the document, so please create a title that sufficiently contains the information in this document.
          For reference, the url of this document is {url}. Use that information if it helps you come up with a title.""",
            },
            {"role": "user", "content": txt},
        ]
    )


def generate_folder_name(txts: list[str]) -> str:
    return api.call_chat(
        [
            {
                "role": "system",
                "content": """"You must name a folder that will contain documents with the titles user going to give you.
         The most important thing is that you must be able to understand the characteristics of the files in the folder just by looking at it, 
         and you must be able to distinguish them from files in other folders. 
         Express it compactly in below three words.""",
            },
            {"role": "user", "content": str(txts)},
        ]
    )
