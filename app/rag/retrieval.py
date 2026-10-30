"""Semantic candidates bypass lexical gates, never source or authorization checks."""
from dataclasses import dataclass
from functools import lru_cache
import json
import re

from rank_bm25 import BM25Okapi

from app.audit import current_request_id
from app.tools.retrieval import _tokenize, policy_topic_matches, contains_policy_override
from .config import Settings
from .documents import (allowed, digest, load_sources, sections, split_sources,
                        verify_governance, verify_span)
from .embeddings import LocalEmbeddings, validate_vectors
from .errors import SourceUnauthorized, PolicyConflict, IndexCorrupt, ModelMismatch
from .index import checked_index


class PolicySnippets(list):
    def __init__(self, passages, metadata):
        super().__init__(passages)
        self.retrieval_metadata = metadata


@dataclass
class RetrievalResult:
    evidence: list[dict]
    passages: list[str]
    metadata: dict


# Source headings define global qualifications; no question-to-document maps.
_GOVERNING = re.compile(r'eligib|exception|condition|scope|job protection|do not|non.retaliation|coordination', re.I)


def expand_evidence(hits, documents, settings, user_context):
    by_id = {doc.metadata['document_id']: doc for doc in documents}
    spans = {}
    for hit in hits:
        meta = hit['metadata']
        doc = by_id[meta['document_id']]
        all_sections = sections(doc)
        for n, (start, end, heading) in enumerate(all_sections):
            selected = start == meta['section_start']
            governing = n == 0 or _GOVERNING.search(heading) or re.search(r'\b(?:unless|except|subject to)\b', doc.page_content[start:end], re.I)
            if not (selected or governing):
                continue
            key = (meta['document_id'], start, end)
            span_meta = {**meta, 'span_start': start, 'span_end': end,
                         'section_heading': heading, 'context_id': 'context:' + digest(str(key)),
                         'context_role': 'selected_section' if selected else 'governing_section'}
            passage = doc.page_content[start:end]
            verify_span(settings, span_meta, passage, user_context)
            if contains_policy_override(passage):
                raise PolicyConflict()
            if key not in spans:
                spans[key] = (passage, span_meta)
    return list(spans.values())


class RetrievalService:
    def __init__(self, settings, provider=None):
        self.settings = settings
        self.provider = provider

    def retrieve(self, question, user_context, top_k=5, mode='hybrid'):
        if mode not in {'lexical', 'semantic', 'hybrid'}:
            raise ValueError('Invalid retrieval mode')
        if not 1 <= top_k <= 20:
            raise ValueError('top_k must be between 1 and 20')
        documents = load_sources(self.settings)
        if mode == 'lexical':
            manifest = {'collection': 'lexical', 'embedding_version': 'none'}
            chunks = split_sources(documents, self.settings, 'none')
            rows = {c.metadata['chunk_id']: (c.page_content, c.metadata, None) for c in chunks}
            semantic = []
        else:
            self.provider = self.provider or LocalEmbeddings(self.settings)
            manifest, store, rows, documents = checked_index(self.settings, self.provider, documents)
            vector = self.provider.embed_query(question)
            validate_vectors([vector], 1, manifest['dimensions'])
            if self.provider.identity() != manifest['embedding_version']:
                raise ModelMismatch()
            try:
                matches = store.similarity_search_by_vector_with_relevance_scores(vector, k=len(rows))
            except Exception:
                raise IndexCorrupt() from None
            semantic = []
            for document, distance in matches:
                cid = document.metadata.get('chunk_id')
                if cid not in rows or document.page_content != rows[cid][0] or document.metadata != rows[cid][1]:
                    raise IndexCorrupt()
                similarity = 1 - float(distance)
                if similarity >= self.settings.min_similarity:
                    semantic.append((cid, similarity))
        full = {d.metadata['document_id']: d.page_content for d in documents}
        lexical = []
        tokens = _tokenize(question)
        ids = sorted(rows)
        if tokens and mode != 'semantic':
            corpus = [_tokenize(rows[cid][1]['document_title'] + '\n' + rows[cid][0]) for cid in ids]
            scores = BM25Okapi(corpus).get_scores(tokens)
            lexical = [(cid, float(score)) for cid, score in zip(ids, scores)
                       if score > 0 and policy_topic_matches(question, full[rows[cid][1]['document_id']])]
            lexical.sort(key=lambda item: (-item[1], item[0]))
        # Authorization is checked before text, titles or rankings reach the caller.
        relevant = set(cid for cid, _ in semantic + lexical)
        denied = {cid for cid in relevant if not allowed(rows[cid][1], user_context)}
        permitted = relevant - denied
        if denied and not permitted:
            raise SourceUnauthorized()
        lexical = [(cid, s) for cid, s in lexical if cid in permitted]
        semantic = [(cid, s) for cid, s in semantic if cid in permitted]
        ranks, scores = {}, {}
        paths = [('lexical', lexical)] if mode == 'lexical' else [('semantic', semantic)] if mode == 'semantic' else [('lexical', lexical), ('semantic', semantic)]
        for method, ranking in paths:
            for rank, (cid, score) in enumerate(ranking, 1):
                ranks.setdefault(cid, {})[method] = rank
                scores.setdefault(cid, {})[method] = score
        order = sorted(ranks, key=lambda cid: (-sum(1 / (60 + rank) for rank in ranks[cid].values()), cid))
        hits = []
        for rank, cid in enumerate(order[:top_k], 1):
            text, metadata, _ = rows[cid]
            verify_span(self.settings, metadata, text, user_context)
            if contains_policy_override(text):
                raise PolicyConflict()
            hits.append({'chunk_id': cid, 'document_id': metadata['document_id'],
                'section_title': metadata['section_heading'], 'passage': text, 'metadata': metadata,
                'retrieval_method': mode, 'rank': rank, 'scores': scores[cid],
                'component_ranks': ranks[cid], 'rrf_score': sum(1 / (60 + r) for r in ranks[cid].values()),
                'source_verification': 'matched'})
        expanded = expand_evidence(hits, documents, self.settings, user_context)
        diagnostics = {'schema_version': 'atlas.retrieval.v1', 'request_id': current_request_id.get(),
            'mode': mode, 'embedding_model': self.settings.model if mode != 'lexical' else None,
            'embedding_version': manifest['embedding_version'], 'index_version': manifest['collection'],
            'outcome': 'evidence_available' if hits else 'no_relevant_evidence',
            'context_omitted_due_to_budget': False,
            'candidates': [{k: v for k, v in hit.items() if k not in {'passage', 'metadata'}} |
                {'source_file': hit['metadata']['source_file'], 'document_title': hit['metadata']['document_title']}
                for hit in hits],
            'context_spans': [meta for _, meta in expanded]}
        return RetrievalResult(hits, [text for text, _ in expanded], diagnostics)


@lru_cache(maxsize=4)
def get_service(settings=None):
    return RetrievalService(settings or Settings())


def verified_citation(quote, snippets, retrieval, settings, user_context):
    from app.diagnostics import OutputRuleError
    spans = retrieval.get('context_spans', [])
    if len(spans) != len(snippets):
        raise OutputRuleError('citation_unverified', 'Citation lacks verified retrieval metadata')
    matches = {}
    for passage, metadata in zip(snippets, spans):
        if quote not in passage:
            continue
        source = verify_span(settings, metadata, passage, user_context)
        start = metadata['span_start'] + passage.index(quote)
        key = (metadata['document_id'], start)
        matches[key] = (source, metadata, start)
    if not matches:
        raise OutputRuleError('citation_absent', 'Citation is absent from selected passages')
    if len(matches) != 1:
        raise OutputRuleError('citation_ambiguous', 'Citation has multiple exact source locations')
    source, meta, start = next(iter(matches.values()))
    return {'source_id': 'sha256:' + digest(source), 'source_file': meta['source_path'],
        'source_verification': 'matched', 'policy_id': meta['document_id'],
        'policy_version': meta.get('policy_version'), 'effective_date': meta.get('effective_date'),
        'approval_status': meta.get('approval_status', 'unknown'), 'applicability': 'unverified',
        'chunk_id': meta['chunk_id'], 'index_version': retrieval['index_version'],
        'quote': quote, 'start': start, 'end': start + len(quote)}


def guard_deterministic_sources(intent, user_context):
    """Known routes still obey authoritative source ACLs without needing vectors."""
    settings = Settings()
    filename = {'PTO_POLICY': 'pto-policy.md', 'REMOTE_WORK': 'remote-work-policy.md'}[intent]
    for doc in load_sources(settings):
        if doc.metadata['source_path'] == filename:
            if not allowed(doc.metadata, user_context):
                raise SourceUnauthorized()
            verify_governance(doc.metadata)
