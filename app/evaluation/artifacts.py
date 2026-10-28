"""Private, content-minimizing artifacts. No raw question/answer/source retention."""
import json
import os
from pathlib import Path
import stat
import tempfile
import time

class ArtifactStore:
    def __init__(self,path):
        self.path=Path(path)
        if self.path.is_symlink():raise ValueError('Symlink artifact root forbidden')
        self.path.mkdir(mode=0o700,parents=True,exist_ok=True)
        s=self.path.stat()
        if s.st_uid!=os.getuid() or stat.S_IMODE(s.st_mode)&0o077:
            raise ValueError('Artifact directory must be owner-only (0700)')

    def write(self,name,data):
        if Path(name).name!=name or not name.endswith('.json'):raise ValueError('Invalid artifact name')
        target=self.path/name
        if target.is_symlink():raise ValueError('Symlink artifact forbidden')
        fd,temp=tempfile.mkstemp(dir=self.path,prefix='.write-')
        try:
            with os.fdopen(fd,'w') as f:json.dump(data,f,indent=2,allow_nan=False)
            os.replace(temp,target)
        finally:
            if os.path.exists(temp):os.unlink(temp)

    def prune(self,days=7):
        if not 1<=days<=30:raise ValueError('Retention must be 1–30 days')
        count=0
        for p in self.path.glob('*.json'):
            if p.is_symlink():continue
            if p.stat().st_mtime<time.time()-days*86400:p.unlink();count+=1
        return count
