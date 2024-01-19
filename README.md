# bookmark-cleaner

A project that organizes the URL list according to the Chrome bookmark format according to Chrome's bookmark function. The functions are as follows.

1. The name is determined according to the document content of the url.
2. Create folders for URLs according to the document content and assign them to folders by topic.
3. Name the folder in 2 to represent the URLs in it.

## Use Example
```python
from bookmark_cleaner import url
from bookmark_cleaner.cleaner import Cleaner

results = Cleaner(urls=url.files).cleanse()
print([result.model_dump() for result in results])
```