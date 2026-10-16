import re


class FineWebCleaner:
    """
    Source-specific cleaning for FineWeb-Edu.

    FineWeb-Edu is already heavily processed,
    so we keep this cleaner intentionally conservative.
    """

    def clean(self, text):
        return text


class WikipediaCleaner:
    """
    Source-specific cleaning for Wikipedia.

    Wikipedia text is already fairly clean.
    We preserve headings, paragraphs, numbers,
    punctuation, and scientific notation.
    """

    def clean(self, text):
        return text


class GutenbergCleaner:
    """
    Remove common Project Gutenberg boilerplate
    while preserving the actual book content.
    """

    START_PATTERNS = [
        r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\*\s*START OF PROJECT GUTENBERG EBOOK.*?\*\*\*",
    ]

    END_PATTERNS = [
        r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        r"\*\*\*\s*END OF PROJECT GUTENBERG EBOOK.*?\*\*\*",
    ]

    def remove_gutenberg_boundaries(self, text):
        """
        Remove text before the Gutenberg START marker
        and after the Gutenberg END marker when present.
        """

        for pattern in self.START_PATTERNS:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )

            if match:
                text = text[match.end():]
                break

        for pattern in self.END_PATTERNS:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE | re.DOTALL,
            )

            if match:
                text = text[:match.start()]
                break

        return text
    
    def remove_legacy_metadata_lines(self, text):
        """
        Remove obvious Gutenberg metadata lines from older files.
        """

        lines = text.split("\n")

        cleaned_lines = []

        patterns = [
            r"^\s*The Project Gutenberg Etext of.*$",
            r"^\s*The Project Gutenberg EBook of.*$",
            r"^\s*Project Gutenberg EBook of.*$",
            r"^\s*Release Date:.*$",
            r"^\s*Language:.*$",
            r"^\s*Character set encoding:.*$",
            r"^\s*Produced by:.*$",
            r"^\s*\[Etext #\d+\].*$",
        ]

        for line in lines:
            should_remove = False

            for pattern in patterns:
                if re.match(
                    pattern,
                    line,
                    flags=re.IGNORECASE,
                ):
                    should_remove = True
                    break

            if not should_remove:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def clean(self, text):
        text = self.remove_gutenberg_boundaries(text)
        text = self.remove_legacy_metadata_lines(text)

        return text.strip()