from functools import lru_cache

import fastapi
import pydantic
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import torch
import platform
import pickle
from annoy import AnnoyIndex
import numpy as np
from sklearn.decomposition import PCA



app = fastapi.FastAPI(
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

VECTOR_SIZE = 768
N_TREES = 50
MODEL = "sentence-transformers/all-mpnet-base-v2"

EmbVector = pydantic.conlist(float, min_length=VECTOR_SIZE, max_length=VECTOR_SIZE)
VolumeVector = pydantic.conlist(float, min_length=3, max_length=3)

Word2Vec = dict[str, np.ndarray]
Word2Index = dict[str, int]
Index2Word = dict[int, str]
Distance = float

CENTER = np.zeros(VECTOR_SIZE)


def get_device() -> torch.device:
    if platform.system() == "Darwin":
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def load_model() -> (
    tuple[SentenceTransformer, Word2Vec, Word2Index, Index2Word, AnnoyIndex]
):
    model = SentenceTransformer(MODEL, trust_remote_code=True).to(get_device())

    with open("word2vec2.pkl", "rb") as f:
        word2vec = pickle.load(f)
    word2index = {word: i for i, word in enumerate(word2vec.keys())}
    index2word = {i: word for i, word in enumerate(word2vec.keys())}

    index = AnnoyIndex(VECTOR_SIZE, "angular")
    index.load("word2vec2.ann")

    return model, word2vec, word2index, index2word, index


model, word2vec, word2index, index2word, index = load_model()


def get_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def _vectorify(*strings: str) -> list[np.ndarray]:
    strings = list(strings)
    embeddings = model.encode(strings, convert_to_tensor=True)
    embeddings = F.normalize(embeddings, p=2, dim=1)
    embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)

    return [emb.cpu().numpy() for emb in embeddings]


@lru_cache(maxsize=256)
def vectorify(word: str) -> np.ndarray:
    return _vectorify(word)[0]


def get_coordinates(word):
    return word2vec[word] if word in word2vec else vectorify(word)


def get_nearest_neighbors(word, n=10):
    emb = get_coordinates(word) if isinstance(word, str) else word
    indices, distances = index.get_nns_by_vector(emb, n, include_distances=True)
    return [
        (
            index2word[index],
            distance,
        )
        for index, distance in zip(indices, distances)
    ]


class SemanticAxis(pydantic.BaseModel):
    start: str
    end: str
    shift: float = 0.5


class SearchRequest(pydantic.BaseModel):
    axes: list[SemanticAxis]
    n: int = 10


class Word(pydantic.BaseModel):
    text: str
    frequency: float
    projection: VolumeVector | None = None
    distance: float | None = None

    


class SearchResponse(pydantic.BaseModel):
    words: list[Word]


def project(points, d=3):
    """
    Project points onto the best fitting plane using PCA.

    Parameters:
    - points (np.array): A NumPy array with shape (n_samples, n_features)
                        where n_samples is the number of data points and
                        n_features is the dimensionality of each data point.
    - d (int): The number of dimensions to project the points onto.

    Returns:
    - np.array: The projection of the points onto the best fitting plane.
    """
    return PCA(d).fit_transform(points)


@app.post("/api/search")
def search(search: SearchRequest) -> SearchResponse:
    cursor = CENTER

    search_words = set()
    for axis in search.axes:
        start = get_coordinates(axis.start)
        end = get_coordinates(axis.end)
        shift = axis.shift
        cursor += (end - start) * shift
        search_words |= {axis.start, axis.end}

    neigbours = get_nearest_neighbors(cursor, search.n)
    new_words = {w for w, _ in neigbours}

    words: list[dict] = [
        dict(
            text=w,
            vector=get_coordinates(w),
            distance=distance,
        )
        for w, distance in neigbours
    ] + [
        dict(
            text=w,
            vector=(vector := get_coordinates(w)),
            distance=get_distance(cursor, vector).tolist(),
        )
        for w in search_words - new_words
    ]

    projections = project([word['vector'] for word in words], d=3)
    projections = projections - projections.mean(axis=0)
    projections = projections / np.abs(projections).max()

    return SearchResponse(words=[
        Word(
            text=word["text"],
            frequency=word["distance"],
            projection=projection.tolist(),
            distance=word["distance"],
        )
        for word, projection in zip(words, projections)
    ])
