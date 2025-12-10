"""
Generate CLI reference documentation from argparse definitions.

Inspired by vLLM's approach: automatically extracts CLI arguments, types,
defaults, and help text from the CLI module's argparse configuration.
"""

import argparse
import logging
import sys
from argparse import SUPPRESS, Action, HelpFormatter
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

logger = logging.getLogger("mkdocs")

ROOT_DIR = Path(__file__).parent.parent.parent
ARGPARSE_DOC_DIR = ROOT_DIR / "docs" / "argparse"


class MarkdownFormatter(HelpFormatter):
    """Custom formatter that generates markdown for argument groups.

    Based on vLLM's MarkdownFormatter for consistent documentation.
    """

    def __init__(self, prog: str, starting_heading_level: int = 3):
        super().__init__(prog, max_help_position=sys.maxsize, width=sys.maxsize)

        self._section_heading_prefix = "#" * starting_heading_level
        self._argument_heading_prefix = "#" * (starting_heading_level + 1)
        self._markdown_output = []

    def start_section(self, heading: str):
        if heading not in {"positional arguments", "options"}:
            heading_md = f"\n{self._section_heading_prefix} {heading}\n\n"
            self._markdown_output.append(heading_md)

    def end_section(self):
        pass

    def add_text(self, text: str):
        if text:
            self._markdown_output.append(f"{text.strip()}\n\n")

    def add_usage(self, usage, actions, groups, prefix=None):
        pass

    def add_arguments(self, actions: Iterable[Action]):
        for action in actions:
            # Skip help and positional arguments
            if len(action.option_strings) == 0:
                # Handle positional arguments differently
                if action.dest != "help":
                    heading_md = f"{self._argument_heading_prefix} `{action.dest.upper()}`\n\n"
                    self._markdown_output.append(heading_md)

                    if action.help:
                        self._markdown_output.append(f"{action.help}\n\n")

                    if action.choices:
                        choices = f"`{'`, `'.join(str(c) for c in action.choices)}`"
                        self._markdown_output.append(f"Possible choices: {choices}\n\n")
                continue

            if "--help" in action.option_strings:
                continue

            option_strings = f"`{'`, `'.join(action.option_strings)}`"
            heading_md = f"{self._argument_heading_prefix} {option_strings}\n\n"
            self._markdown_output.append(heading_md)

            if choices := action.choices:
                choices = f"`{'`, `'.join(str(c) for c in choices)}`"
                self._markdown_output.append(f"Possible choices: {choices}\n\n")
            elif (metavar := action.metavar) and isinstance(metavar, (list, tuple)):
                metavar = f"`{'`, `'.join(str(m) for m in metavar)}`"
                self._markdown_output.append(f"Possible choices: {metavar}\n\n")

            if action.help:
                self._markdown_output.append(f"{action.help}\n\n")

            if (default := action.default) != SUPPRESS:
                # Make empty string defaults visible
                if default == "":
                    default = '""'
                # Skip None defaults for optional arguments
                if default is not None:
                    self._markdown_output.append(f"Default: `{default}`\n\n")

    def format_help(self):
        """Return the formatted help as markdown."""
        return "".join(self._markdown_output)


def create_markdown_parser(parser: argparse.ArgumentParser) -> str:
    """Create markdown documentation for a parser.

    Args:
        parser: The ArgumentParser to document

    Returns:
        Markdown string with formatted arguments
    """
    # Create a new formatter
    formatter = MarkdownFormatter(parser.prog)

    # Format the actions
    formatter.add_arguments(parser._actions)

    return formatter.format_help()


def on_startup(command: Literal["build", "gh-deploy", "serve"], dirty: bool):
    """Generate CLI reference documentation on build."""
    generate_cli_argparse_docs()


def generate_cli_argparse_docs():
    """Generate CLI argparse documentation files."""
    logger.info("Generating CLI argparse documentation...")

    # Create the ARGPARSE_DOC_DIR if it doesn't exist
    if not ARGPARSE_DOC_DIR.exists():
        ARGPARSE_DOC_DIR.mkdir(parents=True)

    try:
        # Import CLI module to access argparse
        sys.path.insert(0, str(ROOT_DIR))
        from supertonic.cli import create_parser

        # Create main parser
        parser = create_parser()

        # Find subparsers
        subparsers_actions = [
            action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
        ]

        if not subparsers_actions:
            logger.warning("No subparsers found in CLI")
            return

        # Generate documentation for each subcommand
        for subparsers_action in subparsers_actions:
            for name, subparser in subparsers_action.choices.items():
                # Skip aliases (we only want main command names)
                # Check if this is an alias by looking at the parent
                if name in ["t", "s", "lv", "i", "d", "v", "synthesize"]:
                    continue

                doc_path = ARGPARSE_DOC_DIR / f"{name}.inc.md"
                markdown = create_markdown_parser(subparser)

                # Write to file
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write(markdown)

                logger.info(f"✓ Generated {doc_path.relative_to(ROOT_DIR)}")
                print(f"✓ Generated {doc_path.relative_to(ROOT_DIR)}")

    except ImportError as e:
        logger.error(f"Failed to import CLI module: {e}")
        logger.warning("CLI argparse documentation generation skipped")
        print(f"❌ Failed to import CLI module: {e}")
    except Exception as e:
        logger.error(f"Failed to generate CLI argparse documentation: {e}")
        logger.warning("Using existing argparse files if available")
        print(f"❌ Failed to generate: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO)
    generate_cli_argparse_docs()
