"""
Generate API reference documentation from source code.

Automatically creates markdown stubs for mkdocstrings based on the
supertonic package structure.
"""

import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger("mkdocs")

ROOT_DIR = Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / "supertonic"
API_DIR = ROOT_DIR / "docs" / "api"


def generate_api_docs():
    """Generate API reference documentation for all modules."""

    # Create API directory
    API_DIR.mkdir(parents=True, exist_ok=True)

    # Module configurations - these will generate detailed module pages
    modules = ["pipeline", "core", "loader", "utils", "config", "cli"]

    # index.md is manually maintained (vLLM style with autorefs links)
    # No need to auto-generate it

    # Generate individual module pages
    for module_name in modules:
        # Show module with full path as title
        content = f"""# supertonic.{module_name}

::: supertonic.{module_name}
    options:
      show_root_heading: true
      show_root_full_path: true
      show_source: true
      heading_level: 2
      members_order: source
      show_signature_annotations: true
      separate_signature: true
      show_symbol_type_heading: true
      show_symbol_type_toc: true
      group_by_category: false
      show_category_heading: false
      docstring_section_style: table
      show_docstring_examples: true
      show_docstring_attributes: true
      show_docstring_functions: true
      show_docstring_classes: true
      show_docstring_modules: true
      merge_init_into_class: true
      show_bases: true
      show_if_no_docstring: true
      filters:
        - "!^_"
        - "!^__"
"""

        with open(API_DIR / f"{module_name}.md", "w", encoding="utf-8") as f:
            f.write(content)

    logger.info(f"✓ Generated {len(modules)} API reference pages")


def on_startup(command: Literal["build", "gh-deploy", "serve"], dirty: bool):
    """Generate API reference documentation on build."""
    logger.info("Generating API reference documentation...")

    try:
        generate_api_docs()
    except Exception as e:
        logger.error(f"Failed to generate API reference: {e}")
        raise
