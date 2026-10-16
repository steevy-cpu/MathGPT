import json
from pathlib import Path


class ByteTokenizer:
    """
    Simple byte-level tokenizer.

    This tokenizer does NOT use BPE yet.
    Each UTF-8 byte becomes one token.
    """

    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config = self.load_config()

        self.pad_token_id = self.config[
            "special_token_ids"
        ]["pad_token_id"]

        self.bos_token_id = self.config[
            "special_token_ids"
        ]["bos_token_id"]

        self.eos_token_id = self.config[
            "special_token_ids"
        ]["eos_token_id"]

        # Byte token IDs begin after the special tokens.
        self.byte_offset = 3

        # There are exactly 256 possible byte values.
        self.byte_vocab_size = 256

        self.total_vocab_size = (
            self.byte_offset
            + self.byte_vocab_size
        )

    def load_config(self):
        """
        Load tokenizer configuration.
        """

        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def encode(
        self,
        text,
        add_bos=False,
        add_eos=False,
    ):
        """
        Convert text into byte token IDs.
        """

        if not isinstance(text, str):
            raise TypeError(
                "ByteTokenizer.encode() expects a string."
            )

        byte_values = text.encode("utf-8")

        token_ids = [
            byte_value + self.byte_offset
            for byte_value in byte_values
        ]

        if add_bos:
            token_ids.insert(
                0,
                self.bos_token_id,
            )

        if add_eos:
            token_ids.append(
                self.eos_token_id
            )

        return token_ids

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
    ):
        """
        Convert byte token IDs back into text.
        """

        byte_values = []

        special_token_ids = {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
        }

        for token_id in token_ids:

            if token_id in special_token_ids:

                if skip_special_tokens:
                    continue

                raise ValueError(
                    "Cannot directly decode special tokens "
                    "into UTF-8 text."
                )

            byte_value = (
                token_id
                - self.byte_offset
            )

            if not 0 <= byte_value <= 255:
                raise ValueError(
                    f"Invalid byte token ID: {token_id}"
                )

            byte_values.append(
                byte_value
            )

        byte_sequence = bytes(
            byte_values
        )

        return byte_sequence.decode(
            "utf-8"
        )