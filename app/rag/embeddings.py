"""Bounded local LangChain embeddings with explicit identity and vector checks."""
from concurrent.futures import ThreadPoolExecutor
import math
import threading
import re

import requests
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from .errors import EmbeddingUnavailable, EmbeddingInvalid, ModelMismatch


class StrictOllamaEmbeddings(OllamaEmbeddings):
    """Pinned LC 1.1.0 adapter with Ollama's explicit no-truncation option.

    LC's current public embed_documents signature omits truncate. This small
    override retains its client/options, but makes oversized inputs fail.
    """
    def embed_documents(self, texts):
        return self._client.embed(self.model, texts, truncate=False,
            dimensions=self.dimensions, options=self._default_params,
            keep_alive=self.keep_alive)['embeddings']

def validate_vectors(vectors, count, dimensions=None):
    if not isinstance(vectors, list) or len(vectors) != count:
        raise EmbeddingInvalid()
    for vector in vectors:
        if not isinstance(vector, list) or not vector:
            raise EmbeddingInvalid()
        dimensions = dimensions or len(vector)
        if len(vector) != dimensions or not all(type(x) in (int, float) and math.isfinite(x) for x in vector) or not any(vector):
            raise EmbeddingInvalid()
    return dimensions


class LocalEmbeddings(Embeddings):
    def __init__(self, settings):
        self.settings = settings
        self._limit = threading.BoundedSemaphore(settings.concurrency)
        self.adapter = StrictOllamaEmbeddings(model=settings.model, base_url=settings.base_url,
            client_kwargs={'timeout': settings.timeout, 'trust_env': False})

    def identity(self):
        try:
            with requests.Session() as session:
                session.trust_env = False
                response = session.get(self.settings.base_url.rstrip('/') + '/api/tags', timeout=self.settings.timeout)
                response.raise_for_status()
                models = response.json()['models']
            name = self.settings.model
            aliases = {name, name + ':latest'} if ':' not in name else {name}
            found = {m['digest'] for m in models if m.get('name') in aliases or m.get('model') in aliases}
            if len(found) != 1:
                raise ModelMismatch()
            version = found.pop()
            if not re.fullmatch(r'[a-f0-9]{64}', version):
                raise ModelMismatch()
            return version
        except (requests.RequestException, ValueError, KeyError, TypeError):
            raise EmbeddingUnavailable() from None

    def embed_documents(self, texts):
        return self._embed([self.settings.document_prefix + text for text in texts])

    def _embed(self, texts):
        with self._limit:
            try:
                vectors = self.adapter.embed_documents(texts)
            except Exception:
                # Never expose service exception strings, request bodies, or URLs.
                raise EmbeddingUnavailable() from None
        validate_vectors(vectors, len(texts))
        return vectors

    def embed_query(self, text):
        return self._embed([self.settings.query_prefix + text])[0]


def embed_batches(provider, texts, settings, dimensions=None):
    batches = [texts[i:i + settings.batch_size] for i in range(0, len(texts), settings.batch_size)]
    vectors = []
    # executor.map keeps input order; both workers and HTTP calls are bounded.
    with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
        for batch, output in zip(batches, pool.map(provider.embed_documents, batches)):
            dimensions = validate_vectors(output, len(batch), dimensions)
            vectors.extend(output)
    return vectors, dimensions
