import json
from pathlib import Path
from collections import Counter


class BPETrainer:
    """
    Train Byte Pair Encoding merges from byte token sequences.

    This trainer starts from the 256 UTF-8 byte tokens
    produced by ByteTokenizer and learns larger tokens
    by repeatedly merging the most common adjacent pair.
    """

    def __init__(
        self,
        byte_offset=3,
        num_merges=20,
    ):
        self.byte_offset = byte_offset
        self.num_merges = num_merges

        # Byte tokenizer vocabulary:
        # IDs 0,1,2 = special tokens
        # IDs 3...258 = raw bytes
        self.base_vocab_size = (
            byte_offset + 256
        )

        self.next_token_id = (
            self.base_vocab_size
        )

        # Learned merges will be stored here.
        self.merges = []

        # Map token IDs back to the raw bytes
        # they represent.
        self.token_bytes = {}

        self.initialize_byte_vocabulary()

    def initialize_byte_vocabulary(self):
        """
        Create the original 256 byte tokens.
        """

        for byte_value in range(256):

            token_id = (
                byte_value
                + self.byte_offset
            )

            self.token_bytes[token_id] = bytes(
                [byte_value]
            )

    def text_to_tokens(self, text):
        """
        Convert text to the initial byte token IDs.
        """

        byte_values = text.encode(
            "utf-8"
        )

        return [
            byte_value + self.byte_offset
            for byte_value in byte_values
        ]

    def count_pairs(self, sequences):
        """
        Count adjacent token pairs across all sequences.
        """

        pair_counts = Counter()

        for sequence in sequences:

            for index in range(
                len(sequence) - 1
            ):

                pair = (
                    sequence[index],
                    sequence[index + 1],
                )

                pair_counts[pair] += 1

        return pair_counts

    def choose_best_pair(self, pair_counts):
        """
        Select the most frequent pair without sorting
        the entire pair-frequency table.

        Ties are resolved deterministically by choosing
        the pair with the smallest token IDs.
        
        complexity: o(P) linear
        """

        if not pair_counts:
            return None, 0

        best_pair = None
        best_count = -1

        for pair, count in pair_counts.items():

            if count > best_count:

                best_pair = pair
                best_count = count

            elif (
                count == best_count
                and pair < best_pair
            ):

                best_pair = pair

        return best_pair, best_count

    def merge_pair(
        self,
        sequence,
        pair,
        new_token_id,
    ):
        """
        Replace occurrences of one token pair
        with a newly created token.
        """

        merged_sequence = []

        index = 0

        while index < len(sequence):

            if (
                index < len(sequence) - 1
                and sequence[index] == pair[0]
                and sequence[index + 1] == pair[1]
            ):

                merged_sequence.append(
                    new_token_id
                )

                index += 2

            else:

                merged_sequence.append(
                    sequence[index]
                )

                index += 1

        return merged_sequence

    def token_to_text(self, token_id):
        """
        Display what a token represents.
        """

        token_bytes = self.token_bytes[
            token_id
        ]

        return token_bytes.decode(
            "utf-8",
            errors="replace",
        )
        
    def save(
        self,
        output_directory,
        tokenizer_config=None,
        ):
        """
        Save learned BPE merges and vocabulary.
        """

        output_directory = Path(
            output_directory
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        merges_path = (
            output_directory
            / "merges.json"
        )

        vocab_path = (
            output_directory
            / "vocab.json"
        )

        config_path = (
            output_directory
            / "tokenizer_config.json"
        )

        # Save merge rules.
        with open(
            merges_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.merges,
                file,
                indent=4,
            )

        # Convert bytes to lists of integers
        # because JSON cannot store raw bytes.
        serializable_vocab = {
            str(token_id): list(token_bytes)
            for token_id, token_bytes
            in self.token_bytes.items()
        }

        with open(
            vocab_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                serializable_vocab,
                file,
                indent=4,
            )

        if tokenizer_config is not None:

            with open(
                config_path,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    tokenizer_config,
                    file,
                    indent=4,
                )

        print()
        print("Tokenizer artifacts saved.")
        print(f"Merges: {merges_path}")
        print(f"Vocabulary: {vocab_path}")

        if tokenizer_config is not None:
            print(f"Config: {config_path}")

    def train(self, texts):
        """
        Learn BPE merges from a list of strings.
        """

        sequences = [
            self.text_to_tokens(text)
            for text in texts
        ]

        print()
        print("Starting BPE training...")
        print(
            f"Requested merges: "
            f"{self.num_merges:,}"
        )
        print()

        for merge_number in range(
            self.num_merges
        ):

            pair_counts = self.count_pairs(
                sequences
            )

            best_pair, frequency = (
                self.choose_best_pair(
                    pair_counts
                )
            )

            if best_pair is None:
                print(
                    "No more pairs available."
                )
                break

            new_token_id = (
                self.next_token_id
            )

            self.next_token_id += 1

            # The new token represents the bytes
            # of both previous tokens combined.
            self.token_bytes[new_token_id] = (
                self.token_bytes[
                    best_pair[0]
                ]
                + self.token_bytes[
                    best_pair[1]
                ]
            )

            self.merges.append(
                {
                    "left": best_pair[0],
                    "right": best_pair[1],
                    "new_id": new_token_id,
                    "frequency": frequency,
                }
            )

            # Apply this merge everywhere.
            sequences = [
                self.merge_pair(
                    sequence,
                    best_pair,
                    new_token_id,
                )
                for sequence in sequences
            ]

            token_text = self.token_to_text(
                new_token_id
            )

            print(
                f"Merge {merge_number + 1:>2}: "
                f"{best_pair} "
                f"-> {new_token_id} "
                f"| frequency={frequency} "
                f"| token={token_text!r}"
            )

        print()
        print("BPE training complete.")

        print(
            f"Learned merges: "
            f"{len(self.merges):,}"
        )

        print(
            f"Vocabulary size: "
            f"{self.base_vocab_size + len(self.merges):,}"
        )

        return self.merges