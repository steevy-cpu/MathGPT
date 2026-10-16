import json
import re
import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TextCleaner:
    """
    Generic text cleaner for MathGPT datasets.

    This class performs only source-independent cleaning.
    """

    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.rules = self.load_rules()

    def load_rules(self):
        """
        Load cleaning rules from JSON.
        """

        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as file:
            config = json.load(file)

        return config["generic_rules"]

    def normalize_unicode(self, text):
        """
        Normalize Unicode characters using the configured form.
        """

        normalization_form = self.rules.get(
            "unicode_normalization",
            "NFC",
        )

        return unicodedata.normalize(
            normalization_form,
            text,
        )

    def normalize_line_endings(self, text):
        """
        Convert Windows and old Mac line endings to \\n.
        """

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        return text

    def remove_null_characters(self, text):
        """
        Remove null-byte characters.
        """

        return text.replace("\x00", "")

    def remove_control_characters(self, text):
        """
        Remove unwanted control characters while preserving
        normal newlines and tabs.
        """

        cleaned_characters = []

        for character in text:

            if character in ("\n", "\t"):
                cleaned_characters.append(character)
                continue

            category = unicodedata.category(character)

            # Unicode control characters begin with "C".
            if category.startswith("C"):
                continue

            cleaned_characters.append(character)

        return "".join(cleaned_characters)

    def strip_trailing_spaces(self, text):
        """
        Remove spaces and tabs from the end of each line.
        """

        lines = text.split("\n")

        cleaned_lines = [
            line.rstrip()
            for line in lines
        ]

        return "\n".join(cleaned_lines)

    def collapse_blank_lines(self, text):
        """
        Limit consecutive blank lines.

        A value of 2 means at most one fully blank line
        between blocks of text.
        """

        max_blank_lines = self.rules.get(
            "max_consecutive_blank_lines",
            2,
        )

        pattern = (
            r"\n{" +
            str(max_blank_lines + 1) +
            r",}"
        )

        replacement = "\n" * max_blank_lines

        return re.sub(
            pattern,
            replacement,
            text,
        )

    def clean(self, text):
        """
        Apply the generic cleaning pipeline.
        """

        if not isinstance(text, str):
            return ""

        if self.rules.get(
            "unicode_normalization",
        ):
            text = self.normalize_unicode(text)

        if self.rules.get(
            "normalize_line_endings",
            True,
        ):
            text = self.normalize_line_endings(text)

        if self.rules.get(
            "remove_null_characters",
            True,
        ):
            text = self.remove_null_characters(text)

        if self.rules.get(
            "remove_control_characters",
            True,
        ):
            text = self.remove_control_characters(text)

        if self.rules.get(
            "strip_trailing_spaces",
            True,
        ):
            text = self.strip_trailing_spaces(text)

        text = self.collapse_blank_lines(text)

        if self.rules.get(
            "strip_document_edges",
            True,
        ):
            text = text.strip()

        return text