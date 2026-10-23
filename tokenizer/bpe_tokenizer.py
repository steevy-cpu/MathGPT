import heapq
import json

try:
    from .byte_tokenizer import ByteTokenizer
except ImportError:
    from byte_tokenizer import ByteTokenizer


class BPETokenizer:
    """
    Byte-level BPE tokenizer.

    Starts from UTF-8 byte tokens and applies learned
    BPE merges using a priority queue.

    Lower merge rank means higher priority.
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

        # ----------------------------------------------------
        # Fast BPE rule lookup
        #
        # pair -> (rank, new_token_id)
        # ----------------------------------------------------

        self.merge_lookup = {}

        for rank, merge in enumerate(
            self.merges
        ):

            pair = (
                merge["left"],
                merge["right"],
            )

            self.merge_lookup[pair] = (
                rank,
                merge["new_id"],
            )

    # ========================================================
    # Load tokenizer
    # ========================================================

    @classmethod
    def from_files(
        cls,
        config_path,
        merges_path,
        vocab_path,
    ):
        """
        Load a trained tokenizer from artifact files.
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

    # ========================================================
    # Heap helpers
    # ========================================================

    def add_pair_to_heap(
        self,
        heap,
        tokens,
        alive,
        next_index,
        left_index,
    ):
        """
        Look at one adjacent token pair.

        If the pair has a learned BPE rule,
        place it in the priority queue.
        """

        if left_index is None:
            return

        if not alive[left_index]:
            return

        right_index = next_index[
            left_index
        ]

        if right_index is None:
            return

        if not alive[right_index]:
            return

        pair = (
            tokens[left_index],
            tokens[right_index],
        )

        merge_info = (
            self.merge_lookup.get(
                pair
            )
        )

        if merge_info is None:
            return

        rank, new_token_id = (
            merge_info
        )

        # Heap ordering:
        #
        # 1. lower rank first
        # 2. lower left position first
        #
        # The position tie-break preserves
        # left-to-right behavior.
        heapq.heappush(
            heap,
            (
                rank,
                left_index,
                right_index,
                pair[0],
                pair[1],
                new_token_id,
            ),
        )

    # ========================================================
    # Encode
    # ========================================================

    def encode(
        self,
        text,
        add_bos=False,
        add_eos=False,
    ):
        """
        Encode text into BPE token IDs.

        Uses a priority queue and linked-list-style
        neighbor tracking so we do not repeatedly scan
        the entire sequence.
        """

        tokens = (
            self.byte_tokenizer.encode(
                text,
                add_bos=False,
                add_eos=False,
            )
        )

        token_count = len(
            tokens
        )

        if token_count == 0:

            result = []

            if add_bos:
                result.append(
                    self.bos_token_id
                )

            if add_eos:
                result.append(
                    self.eos_token_id
                )

            return result

        # ----------------------------------------------------
        # Linked-list representation
        # ----------------------------------------------------
        #
        # Instead of physically deleting list items every
        # merge, keep neighbor indexes.
        #
        # Example:
        #
        # index:      0   1   2   3
        # token:      t   h   e
        #
        # next:       1   2   3   None
        # previous: None  0   1   2
        #
        # When positions merge, links are rewired.
        # ----------------------------------------------------

        previous_index = [
            index - 1
            if index > 0
            else None
            for index in range(
                token_count
            )
        ]

        next_index = [
            index + 1
            if index < token_count - 1
            else None
            for index in range(
                token_count
            )
        ]

        alive = [
            True
            for _ in range(
                token_count
            )
        ]

        # ----------------------------------------------------
        # Build initial heap
        # ----------------------------------------------------

        heap = []

        for index in range(
            token_count - 1
        ):

            self.add_pair_to_heap(
                heap=heap,
                tokens=tokens,
                alive=alive,
                next_index=next_index,
                left_index=index,
            )

        # ----------------------------------------------------
        # Apply BPE merges
        # ----------------------------------------------------

        while heap:

            (
                rank,
                left,
                right,
                expected_left_token,
                expected_right_token,
                new_token_id,
            ) = heapq.heappop(
                heap
            )

            # ------------------------------------------------
            # Lazy validation
            # ------------------------------------------------
            #
            # Heap entries can become outdated after earlier
            # merges. Instead of removing them eagerly,
            # simply ignore stale entries when popped.
            # ------------------------------------------------

            if (
                not alive[left]
                or not alive[right]
            ):
                continue

            if (
                next_index[left]
                != right
            ):
                continue

            if (
                tokens[left]
                != expected_left_token
                or tokens[right]
                != expected_right_token
            ):
                continue

            # ------------------------------------------------
            # Merge RIGHT into LEFT
            # ------------------------------------------------

            tokens[left] = (
                new_token_id
            )

            alive[right] = False

            # Save neighbors before rewiring.
            previous_left = (
                previous_index[left]
            )

            next_right = (
                next_index[right]
            )

            # LEFT now points past RIGHT.
            next_index[left] = (
                next_right
            )

            if next_right is not None:

                previous_index[
                    next_right
                ] = left

            # RIGHT is dead.
            previous_index[right] = None
            next_index[right] = None

            # ------------------------------------------------
            # Only nearby pairs may have changed.
            #
            # Previous pair:
            #     previous_left + left
            #
            # Next pair:
            #     left + next_right
            # ------------------------------------------------

            if previous_left is not None:

                self.add_pair_to_heap(
                    heap=heap,
                    tokens=tokens,
                    alive=alive,
                    next_index=next_index,
                    left_index=previous_left,
                )

            self.add_pair_to_heap(
                heap=heap,
                tokens=tokens,
                alive=alive,
                next_index=next_index,
                left_index=left,
            )

        # ----------------------------------------------------
        # Reconstruct final token sequence
        # ----------------------------------------------------

        result = []

        index = 0

        while index is not None:

            if alive[index]:

                result.append(
                    tokens[index]
                )

            index = next_index[
                index
            ]

        # ----------------------------------------------------
        # Special tokens
        # ----------------------------------------------------

        if add_bos:

            result.insert(
                0,
                self.bos_token_id,
            )

        if add_eos:

            result.append(
                self.eos_token_id
            )

        return result

    # ========================================================
    # Decode
    # ========================================================

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
    ):
        """
        Decode BPE IDs back into the original UTF-8 text.
        """

        byte_chunks = []

        special_token_ids = {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
        }

        for token_id in token_ids:

            if (
                token_id
                in special_token_ids
            ):

                if skip_special_tokens:
                    continue

                raise ValueError(
                    "Cannot directly decode "
                    "special tokens into UTF-8 text."
                )

            if (
                token_id
                not in self.token_bytes
            ):

                raise ValueError(
                    f"Unknown token ID: "
                    f"{token_id}"
                )

            byte_chunks.append(
                self.token_bytes[
                    token_id
                ]
            )

        byte_sequence = b"".join(
            byte_chunks
        )

        return byte_sequence.decode(
            "utf-8"
        )