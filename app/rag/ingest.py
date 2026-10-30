"""Index configured policy Markdown separately from employee requests."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from dotenv import load_dotenv
from .config import ROOT, Settings
from .errors import RAGError
from .index import ingest


def main():
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docs-dir', type=Path)
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    try:
        settings = Settings()
        if args.docs_dir:
            settings = replace(settings, docs_dir=args.docs_dir.resolve())
        print(json.dumps(ingest(settings, rebuild=args.rebuild), indent=2))
        return 0
    except RAGError as error:
        # No source text, paths, questions or remote exception bodies.
        print(json.dumps({'status': 'failed', 'error_code': error.code,
            'failed_documents': None, 'active_index_replaced': False}), file=sys.stderr)
        return 1
    except ValueError:
        print(json.dumps({'status': 'failed', 'error_code': 'rag_configuration',
            'active_index_replaced': False}), file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
