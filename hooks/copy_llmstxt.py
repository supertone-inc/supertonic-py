"""Publish the repo-root ``llms.txt`` at the built site root.

The canonical ``llms.txt`` for this project lives at the repository root
so it stays visible on GitHub and to repo-aware agents (Cursor, Claude
Code, Copilot Chat) that scan a cloned working tree. This hook also
copies it into the MkDocs build output so the file is served at the
docs-site root (e.g. ``https://supertone-inc.github.io/supertonic-py/llms.txt``)
for web-crawling agents — the location the llmstxt.org spec describes.

There is exactly one source of truth: ``<repo>/llms.txt``. Do not edit a
copy inside ``docs/``.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger("mkdocs")

LLMS_TXT_FILENAME = "llms.txt"


def on_post_build(config) -> None:
    """Copy ``<repo>/llms.txt`` to ``<site_dir>/llms.txt`` after build."""
    repo_root = Path(config["config_file_path"]).parent
    src = repo_root / LLMS_TXT_FILENAME
    if not src.exists():
        logger.warning(
            "%s not found at repo root (%s) — skipping copy.",
            LLMS_TXT_FILENAME,
            repo_root,
        )
        return

    site_dir = Path(config["site_dir"])
    site_dir.mkdir(parents=True, exist_ok=True)
    dst = site_dir / LLMS_TXT_FILENAME
    shutil.copy2(src, dst)

    try:
        rel = dst.relative_to(repo_root)
    except ValueError:
        rel = dst
    logger.info("✓ Published %s → %s", LLMS_TXT_FILENAME, rel)
