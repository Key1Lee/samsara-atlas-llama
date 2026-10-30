"""Copy-on-write Chroma generations; atomic publication preserves active readers."""
from contextlib import contextmanager
from functools import lru_cache
import fcntl
import json
import os
from pathlib import Path
import re
import time
from uuid import uuid4

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from .documents import INDEX_FORMAT, digest, load_sources, split_sources, embedding_text
from .embeddings import LocalEmbeddings, embed_batches, validate_vectors
from .errors import IndexMissing, IndexStale, IndexCorrupt, ModelMismatch, RAGError


def inventory(documents):
    return {d.metadata['source_path']: d.metadata for d in documents}


def read_manifest(settings):
    try:
        value = json.loads((settings.index_dir / 'active.json').read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise IndexMissing() from None
    except (OSError, ValueError):
        raise IndexCorrupt() from None
    if not isinstance(value, dict) or value.get('format') != INDEX_FORMAT or not re.fullmatch(r'atlas_policy_[a-f0-9]{32}', str(value.get('collection', ''))):
        raise IndexCorrupt()
    if (not isinstance(value.get('chunk_ids'), list) or not all(isinstance(cid, str) for cid in value['chunk_ids'])
            or len(set(value['chunk_ids'])) != len(value['chunk_ids'])
            or type(value.get('dimensions')) is not int or value['dimensions'] <= 0
            or not isinstance(value.get('sources'), dict) or not isinstance(value.get('splitting'), dict)
            or not isinstance(value.get('chunks'), dict) or set(value['chunks']) != set(value['chunk_ids'])
            or not isinstance(value.get('model'), str)
            or not re.fullmatch(r'[a-f0-9]{64}', str(value.get('embedding_version', '')))):
        raise IndexCorrupt()
    return value


def splitting_settings(settings):
    return {'chunk_size': settings.chunk_size, 'chunk_overlap': settings.chunk_overlap,
            'document_prefix': settings.document_prefix, 'query_prefix': settings.query_prefix,
            'embedding_input': 'title_and_exact_section.v1'}


@lru_cache(maxsize=8)
def open_store(index_dir, collection):
    try:
        return Chroma(collection_name=collection, persist_directory=str(Path(index_dir) / 'chroma'),
            client_settings=ChromaSettings(anonymized_telemetry=False), create_collection_if_not_exists=False)
    except Exception:
        raise IndexCorrupt() from None


def index_rows(store, manifest):
    try:
        data = store.get(include=['documents', 'metadatas', 'embeddings'])
        if set(data['ids']) != set(manifest['chunk_ids']):
            raise IndexCorrupt()
        rows = {}
        for i, chunk_id in enumerate(data['ids']):
            vector = data['embeddings'][i].tolist()
            validate_vectors([vector], 1, manifest['dimensions'])
            metadata, text = data['metadatas'][i], data['documents'][i]
            if metadata.get('chunk_id') != chunk_id or digest(text) != manifest['chunks'][chunk_id]:
                raise IndexCorrupt()
            rows[chunk_id] = (text, metadata, vector)
        return rows
    except RAGError:
        raise
    except Exception:
        raise IndexCorrupt() from None


def checked_index(settings, provider, documents=None):
    manifest = read_manifest(settings)
    if manifest.get('model') != settings.model or manifest.get('embedding_version') != provider.identity():
        raise ModelMismatch()
    documents = documents if documents is not None else load_sources(settings)
    if manifest.get('sources') != inventory(documents) or manifest.get('splitting') != splitting_settings(settings):
        raise IndexStale()
    store = open_store(str(settings.index_dir), manifest['collection'])
    rows = index_rows(store, manifest)
    expected = split_sources(documents, settings, manifest['embedding_version'])
    if {c.metadata['chunk_id']: digest(c.page_content) for c in expected} != manifest['chunks']:
        raise IndexCorrupt()
    # Metadata and source offsets are checked independently of store-supplied values.
    if any(rows[c.metadata['chunk_id']][1] != c.metadata for c in expected):
        raise IndexCorrupt()
    return manifest, store, rows, documents


@contextmanager
def writer_lock(settings):
    settings.index_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    settings.index_dir.chmod(0o700)
    with (settings.index_dir / 'writer.lock').open('a') as stream:
        os.chmod(stream.name, 0o600)
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RAGError() from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class PreparedEmbeddings(Embeddings):
    """Give LangChain Chroma prevalidated new/reused vectors without re-embedding."""
    def __init__(self, vectors):
        from collections import deque
        vectors = {text: deque(values) for text, values in vectors.items()}
        self.vectors = vectors
    def embed_documents(self, texts):
        return [self.vectors[text].popleft() for text in texts]
    def embed_query(self, text):
        raise RuntimeError('Prepared embeddings are ingestion-only')


def ingest(settings, *, provider=None, rebuild=False):
    started = time.perf_counter()
    provider = provider or LocalEmbeddings(settings)
    with writer_lock(settings):
        documents = load_sources(settings)
        version = provider.identity()
        chunks = split_sources(documents, settings, version)
        old = None
        try:
            old = read_manifest(settings)
        except IndexMissing:
            pass
        if old and not rebuild and (old.get('model') != settings.model or old.get('embedding_version') != version or old.get('splitting') != splitting_settings(settings)):
            raise ModelMismatch()
        reusable = {}
        if old and not rebuild:
            reusable = index_rows(open_store(str(settings.index_dir), old['collection']), old)
        new_chunks = [c for c in chunks if c.metadata['chunk_id'] not in reusable]
        vectors, dimensions = embed_batches(provider, [embedding_text(c) for c in new_chunks], settings,
            old['dimensions'] if old and not rebuild else None)
        if not chunks:
            raise IndexCorrupt()
        dimensions = dimensions or (old or {}).get('dimensions')
        by_id = {c.metadata['chunk_id']: vector for c, vector in zip(new_chunks, vectors)}
        for chunk in chunks:
            cid = chunk.metadata['chunk_id']
            if cid in reusable and cid not in by_id:
                by_id[cid] = reusable[cid][2]
        if provider.identity() != version:
            raise ModelMismatch()
        if inventory(load_sources(settings)) != inventory(documents):
            raise IndexStale()
        changed = sum((old or {}).get('sources', {}).get(name) != meta for name, meta in inventory(documents).items())
        removed = len(set((old or {}).get('sources', {})) - set(inventory(documents)))
        summary = {'documents': len(documents), 'chunks': len(chunks), 'changed_documents': changed,
            'deleted_documents': removed, 'failed_documents': 0, 'embedded_chunks': len(new_chunks),
            'embedding_model': settings.model, 'embedding_version': version, 'dimensions': dimensions}
        if old and not rebuild and not changed and not removed:
            return {**summary, 'index_version': old['collection'], 'duration_seconds': time.perf_counter() - started}
        collection = 'atlas_policy_' + uuid4().hex
        prepared_vectors = {}
        for chunk in chunks:
            prepared_vectors.setdefault(chunk.page_content, []).append(by_id[chunk.metadata['chunk_id']])
        prepared = PreparedEmbeddings(prepared_vectors)
        stage = Chroma(collection_name=collection, persist_directory=str(settings.index_dir / 'chroma'),
            embedding_function=prepared, client_settings=ChromaSettings(anonymized_telemetry=False),
            collection_metadata={'hnsw:space': 'cosine'})
        temporary = settings.index_dir / (collection + '.json.tmp')
        published = False
        try:
            for i in range(0, len(chunks), settings.batch_size):
                batch = chunks[i:i + settings.batch_size]
                stage.add_documents(batch, ids=[c.metadata['chunk_id'] for c in batch])
            manifest = {'format': INDEX_FORMAT, 'collection': collection, 'model': settings.model,
                'embedding_version': version, 'dimensions': dimensions, 'sources': inventory(documents),
                'splitting': splitting_settings(settings), 'chunk_ids': [c.metadata['chunk_id'] for c in chunks],
                'chunks': {c.metadata['chunk_id']: digest(c.page_content) for c in chunks}}
            index_rows(stage, manifest)
            if inventory(load_sources(settings)) != inventory(documents) or provider.identity() != version:
                raise IndexStale()
            with temporary.open('w', encoding='utf-8') as stream:
                json.dump(manifest, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            for path in settings.index_dir.rglob('*'):
                path.chmod(0o700 if path.is_dir() else 0o600)
            os.replace(temporary, settings.index_dir / 'active.json')
            published = True
            # fsync directory makes the pointer rename durable on local POSIX filesystems.
            fd = os.open(settings.index_dir, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception:
            if not published:
                stage.delete_collection()  # Only this unpublished replacement is removed.
            raise
        finally:
            temporary.unlink(missing_ok=True)
        return {**summary, 'index_version': collection, 'duration_seconds': time.perf_counter() - started}
