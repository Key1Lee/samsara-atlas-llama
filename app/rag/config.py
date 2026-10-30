from dataclasses import dataclass, field
from pathlib import Path
import os
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]

def retrieval_mode():
    mode = os.getenv('ATLAS_RETRIEVAL_MODE', 'lexical').lower().strip()
    if mode not in {'lexical', 'semantic', 'hybrid'}:
        raise ValueError('Invalid retrieval mode')
    return mode

@dataclass(frozen=True)
class Settings:
    docs_dir: Path = field(default_factory=lambda: Path(os.getenv('DOCS_DIR', str(ROOT / 'data/docs'))).resolve())
    index_dir: Path = field(default_factory=lambda: Path(os.getenv('ATLAS_RAG_INDEX_DIR', str(ROOT / 'data/rag'))).resolve())
    model: str = field(default_factory=lambda: os.getenv('ATLAS_EMBEDDING_MODEL', 'nomic-embed-text'))
    base_url: str = field(default_factory=lambda: os.getenv('ATLAS_EMBEDDING_URL', 'http://127.0.0.1:11434'))
    chunk_size: int = field(default_factory=lambda: int(os.getenv('ATLAS_RAG_CHUNK_SIZE', '1800')))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv('ATLAS_RAG_CHUNK_OVERLAP', '160')))
    batch_size: int = field(default_factory=lambda: int(os.getenv('ATLAS_RAG_BATCH_SIZE', '8')))
    concurrency: int = field(default_factory=lambda: int(os.getenv('ATLAS_RAG_EMBED_CONCURRENCY', '2')))
    timeout: float = field(default_factory=lambda: float(os.getenv('ATLAS_RAG_EMBED_TIMEOUT', '60')))
    min_similarity: float = field(default_factory=lambda: float(os.getenv('ATLAS_RAG_MIN_SIMILARITY', '0.65')))
    metadata_file: str = field(default_factory=lambda: os.getenv('ATLAS_POLICY_METADATA', ''))
    document_prefix: str | None = field(default_factory=lambda: os.getenv('ATLAS_EMBED_DOCUMENT_PREFIX'))
    query_prefix: str | None = field(default_factory=lambda: os.getenv('ATLAS_EMBED_QUERY_PREFIX'))

    def __post_init__(self):
        # The Nomic model card requires these task prefixes. Other models keep
        # empty prefixes unless the operator explicitly configures them.
        nomic = self.model.split(':')[0] == 'nomic-embed-text'
        if self.document_prefix is None:
            object.__setattr__(self, 'document_prefix', 'search_document: ' if nomic else '')
        if self.query_prefix is None:
            object.__setattr__(self, 'query_prefix', 'search_query: ' if nomic else '')
        url = urlparse(self.base_url)
        if url.scheme != 'http' or url.hostname not in {'localhost', '127.0.0.1', '::1'} or url.username or url.password or url.path not in {'', '/'}:
            raise ValueError('Embeddings require a loopback HTTP Ollama endpoint')
        if not 0 <= self.chunk_overlap < self.chunk_size or self.chunk_size < 100:
            raise ValueError('Invalid chunk bounds')
        if not 1 <= self.batch_size <= 64 or not 1 <= self.concurrency <= 4 or not 0 < self.timeout <= 300:
            raise ValueError('Invalid embedding limits')
        if not -1 <= self.min_similarity <= 1 or not self.model.strip():
            raise ValueError('Invalid embedding configuration')
        if self.index_dir == self.docs_dir or self.index_dir.is_relative_to(self.docs_dir):
            raise ValueError('Index must be outside the policy document directory')
