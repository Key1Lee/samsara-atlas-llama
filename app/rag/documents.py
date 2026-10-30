"""Authorized UTF-8 sources and exact spans; splitter normalization is never cited."""
import hashlib
import json
import re
from datetime import date
from pathlib import Path

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from .errors import SourceUnavailable, SourceUnauthorized, PolicyConflict

INDEX_FORMAT = 'atlas.chroma.v1'

def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def governance(settings):
    if not settings.metadata_file:
        return {}
    try:
        metadata = json.loads(Path(settings.metadata_file).read_text(encoding='utf-8'))
        if not isinstance(metadata, dict):
            raise ValueError()
        for name, values in metadata.items():
            if Path(name).is_absolute() or '..' in Path(name).parts or not isinstance(values, dict):
                raise ValueError()
            if 'access_scope' in values and (not isinstance(values['access_scope'], list) or
                    not all(isinstance(v, str) and v for v in values['access_scope'])):
                raise ValueError()
            for key in ('document_id', 'policy_version', 'approval_status', 'effective_date', 'expires_at'):
                if key in values and (not isinstance(values[key], str) or not values[key]):
                    raise ValueError()
            for key in ('effective_date', 'expires_at'):
                if key in values:
                    date.fromisoformat(values[key])
        return metadata
    except (OSError, UnicodeError, ValueError, TypeError):
        raise SourceUnavailable() from None


def allowed(metadata, user_context):
    scopes = metadata.get('access_scope')
    if isinstance(scopes, str):
        scopes = json.loads(scopes)
    # DOCS_DIR is the configured shared policy boundary, never employee storage.
    # Absence of an authoritative ACL does not fabricate one in indexed metadata.
    return scopes is None or 'all_employees' in scopes or (user_context or {}).get('employee_id') in scopes


def verify_governance(metadata):
    today = date.today()
    if metadata.get('approval_status', 'unknown') not in {'unknown', 'approved'}:
        raise PolicyConflict()
    for key, comparison in (('effective_date', lambda d: d > today), ('expires_at', lambda d: d <= today)):
        if metadata.get(key) and comparison(date.fromisoformat(metadata[key])):
            raise PolicyConflict()


class PolicyMarkdownLoader(BaseLoader):
    def __init__(self, settings):
        self.settings = settings

    def lazy_load(self):
        root = self.settings.docs_dir
        if not root.is_dir() or root.is_symlink():
            raise SourceUnavailable()
        meta = governance(self.settings)
        try:
            # Reject every symlink, including directories rglob would silently skip.
            paths = sorted(root.rglob('*'))
            if any(p.is_symlink() for p in paths):
                raise SourceUnavailable()
            files = [p for p in paths if p.is_file() and p.suffix.lower() == '.md']
            if not files:
                raise SourceUnavailable()
            for path in files:
                if not path.resolve().is_relative_to(root):
                    raise SourceUnavailable()
                name = path.relative_to(root).as_posix()
                # newline='' preserves CRLF; offsets index original Unicode text.
                with path.open(encoding='utf-8', newline='') as stream:
                    text = stream.read()
                if not text.strip():
                    raise SourceUnavailable()
                title = next((m.group(1).strip() for m in re.finditer(r'^#\s+(.+)$', text, re.M)), path.stem)
                values = {k: v for k, v in meta.get(name, {}).items() if k in {
                    'document_id', 'policy_version', 'approval_status', 'effective_date', 'expires_at', 'access_scope'}}
                values.setdefault('document_id', 'doc:' + digest(name))
                values.update(source_file=path.name, source_path=name, document_title=title,
                              content_hash=digest(text))
                if isinstance(values.get('access_scope'), list):
                    values['access_scope'] = json.dumps(values['access_scope'])
                yield Document(page_content=text, metadata=values)
        except (OSError, UnicodeError):
            raise SourceUnavailable() from None


def load_sources(settings):
    docs = PolicyMarkdownLoader(settings).load()
    # A registry for selecting superseded versions is not implemented. Fail closed.
    by_title = {}
    for doc in docs:
        title = doc.metadata['document_title']
        by_title.setdefault(title, set()).add(doc.metadata['content_hash'])
    if any(len(hashes) > 1 for hashes in by_title.values()):
        raise PolicyConflict()
    return docs


def sections(document):
    """Use LC heading splits to locate raw sections without rewriting whitespace."""
    text = document.page_content
    split = MarkdownHeaderTextSplitter(
        headers_to_split_on=[('#' * n, f'h{n}') for n in range(1, 7)], strip_headers=False,
    ).split_text(text)
    starts, cursor = [], 0
    for item in split:
        headers = [(int(k[1:]), v) for k, v in item.metadata.items() if re.fullmatch(r'h[1-6]', k)]
        if headers:
            level, title = max(headers)
            # Search the exact heading line, respecting duplicate headings in order.
            match = re.search(r'^' + '#' * level + r'[ \t]+' + re.escape(title) + r'[ \t]*\r?$', text[cursor:], re.M)
            if match is None:
                raise SourceUnavailable()
            start = cursor + match.start()
            cursor = cursor + match.end()
        else:
            start, title = 0, document.metadata['document_title']
        if not starts or start > starts[-1][0]:
            starts.append((start, title))
    if not starts or starts[0][0] > 0:
        starts.insert(0, (0, document.metadata['document_title']))
    result = []
    for i, (start, heading) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        if text[start:end].strip():
            result.append((start, end, heading))
    return result


def split_sources(documents, settings, embedding_version):
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap, add_start_index=True, strip_whitespace=False,
        separators=['\n\n', '\n', '. ', ' ', ''])
    chunks = []
    for doc in documents:
        for section_start, section_end, heading in sections(doc):
            section_text = doc.page_content[section_start:section_end]
            if not re.sub(r'^#{1,6}[^\n]*(?:\n|$)', '', section_text, flags=re.M).strip():
                continue  # A heading alone is context, not policy evidence.
            for part in splitter.create_documents([section_text]):
                offset = part.metadata['start_index']
                if offset < 0 or section_text[offset:offset + len(part.page_content)] != part.page_content:
                    raise SourceUnavailable()
                start = section_start + offset
                metadata = {**doc.metadata, 'section_heading': heading, 'span_start': start,
                    'span_end': start + len(part.page_content), 'section_start': section_start,
                    'section_end': section_end, 'chunk_index': len(chunks),
                    'embedding_model': settings.model, 'embedding_version': embedding_version,
                    'index_version': INDEX_FORMAT}
                metadata['chunk_id'] = 'chunk:' + digest(doc.metadata['document_id'] + '\0' +
                    str(start) + '\0' + doc.metadata['document_title'] + '\0' + part.page_content)
                chunks.append(Document(page_content=part.page_content, metadata=metadata))
    return chunks


def embedding_text(chunk):
    # Titles disambiguate generic headings such as "Eligibility" without
    # transforming the passage stored/cited in Chroma.
    return chunk.metadata['document_title'] + '\n' + chunk.page_content


def verify_span(settings, metadata, passage, user_context):
    if not allowed(metadata, user_context):
        raise SourceUnauthorized()
    verify_governance(metadata)
    relative = Path(metadata['source_path'])
    root = settings.docs_dir
    path = root / relative
    if relative.is_absolute() or '..' in relative.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise SourceUnavailable()
    if not path.resolve().is_relative_to(root):
        raise SourceUnavailable()
    try:
        with path.open(encoding='utf-8', newline='') as stream:
            source = stream.read()
    except (OSError, UnicodeError):
        raise SourceUnavailable() from None
    from .errors import IndexStale
    if digest(source) != metadata['content_hash'] or source[metadata['span_start']:metadata['span_end']] != passage:
        raise IndexStale()
    return source
