"""Fixed, privacy-safe infrastructure categories, distinct from no evidence."""
class RAGError(RuntimeError):
    code = 'rag_unavailable'
    status_code = 503
    def __init__(self):
        super().__init__(self.code)

class SourceUnavailable(RAGError): code = 'rag_source_unavailable'
class SourceUnauthorized(RAGError):
    code = 'rag_source_unauthorized'
    status_code = 403
class PolicyConflict(RAGError):
    code = 'rag_policy_version_unverified'
    status_code = 409
class IndexMissing(RAGError): code = 'rag_index_missing'
class IndexStale(RAGError): code = 'rag_index_stale'
class IndexCorrupt(RAGError): code = 'rag_index_corrupt'
class ModelMismatch(RAGError): code = 'rag_embedding_mismatch'
class EmbeddingUnavailable(RAGError): code = 'rag_embedding_unavailable'
class EmbeddingInvalid(RAGError): code = 'rag_embedding_invalid'
