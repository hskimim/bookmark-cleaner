import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans  # type: ignore


def cluster_embeds(embeds: npt.NDArray[np.float64], n_clusters: int) -> list[int]:
    cluster = KMeans(n_clusters=n_clusters, n_init="auto").fit(embeds)
    return cluster.predict(embeds)
