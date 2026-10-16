import json

from pathlib import Path

from byte_tokenizer import ByteTokenizer


class BPETokenizer:
    """
    Byte-level BPE tokenizer.

    Starts from the ByteTokenizer representation and applies
    learned BPE merge rules in the exact order they were learned.
    """

    def __init__(
        self,
        config_path,
        merges,
        token_bytes,
    ):
        self.byte_tokenizer = ByteTokenizer(
            config_path
        )

        self.merges = merges

        # Copy the trainer's token-byte mapping.
        self.token_bytes = dict(
            token_bytes
        )

        self.pad_token_id = (
            self.byte_tokenizer.pad_token_id
        )

        self.bos_token_id = (
            self.byte_tokenizer.bos_token_id
        )

        self.eos_token_id = (
            self.byte_tokenizer.eos_token_id
        )

    def merge_pair(
        self,
        token_ids,
        left_id,
        right_id,
        new_id,
    ):
        """
        Apply one BPE merge rule across a token sequence.
        """

        merged = []

        index = 0

        while index < len(token_ids):

            if (
                index < len(token_ids) - 1
                and token_ids[index] == left_id
                and token_ids[index + 1] == right_id
            ):
                merged.append(
                    new_id
                )

                index += 2

            else:
                merged.append(
                    token_ids[index]
                )

                index += 1

        return merged

    def encode(
        self,
        text,
        add_bos=False,
        add_eos=False,
    ):
        """
        Encode text into BPE token IDs.
        """

        # Start from raw byte token IDs.
        token_ids = self.byte_tokenizer.encode(
            text,
            add_bos=False,
            add_eos=False,
        )

        # Apply learned merges in training order.
        for merge in self.merges:

            token_ids = self.merge_pair(
                token_ids,
                merge["left"],
                merge["right"],
                merge["new_id"],
            )

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
    
    @classmethod
    def from_files(
        cls,
        config_path,
        merges_path,
        vocab_path,
        ):
        """
        Load a previously trained BPE tokenizer.
        """

        with open(
            merges_path,
            "r",
            encoding="utf-8",
        ) as file:

            merges = json.load(file)

        with open(
                vocab_path,
                "r",
                encoding="utf-8",
        ) as file:

                saved_vocab = json.load(file)

        token_bytes = {
            int(token_id): bytes(byte_values)
            for token_id, byte_values
            in saved_vocab.items()
        }

        return cls(
            config_path=config_path,
            merges=merges,
            token_bytes=token_bytes,
        )

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
    ):
        """
        Decode BPE token IDs back into text.
        """

        byte_chunks = []

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

            if token_id not in self.token_bytes:
                raise ValueError(
                    f"Unknown token ID: {token_id}"
                )

            byte_chunks.append(
                self.token_bytes[token_id]
            )

        byte_sequence = b"".join(
            byte_chunks
        )

        return byte_sequence.decode(
            "utf-8"
        )