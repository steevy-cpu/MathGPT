from bpe_trainer import BPETrainer


def main():

    training_texts = [
        "the derivative of x is useful",
        "the derivative of y is useful",
        "the derivative helps with calculus",
        "calculus uses the derivative",
        "the matrix is useful",
        "the matrix and the vector",
        "linear algebra uses a matrix",
        "the derivative and the matrix",
    ]

    trainer = BPETrainer(
        num_merges=20
    )

    merges = trainer.train(
        training_texts
    )

    print()
    print("Learned Merge Rules")
    print("===================")

    for merge in merges:

        token_text = trainer.token_to_text(
            merge["new_id"]
        )

        print(
            f"{merge['left']} + "
            f"{merge['right']} "
            f"-> {merge['new_id']} "
            f"= {token_text!r}"
        )


if __name__ == "__main__":
    main()