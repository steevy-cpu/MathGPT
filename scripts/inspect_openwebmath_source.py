from datasets import load_dataset


def main():
    print("Starting OpenWebMath inspection...", flush=True)

    dataset = load_dataset(
        "open-web-math/open-web-math",
        split="train",
        streaming=True, 
    )

    print("Dataset stream created.", flush=True)

    example = next(iter(dataset))

    print("First document received.", flush=True)

    print()
    print("Available fields")
    print("----------------")

    for key in example.keys():
        print(key)

    print()
    print("Field types")
    print("-----------")

    for key, value in example.items():
        print(
            f"{key}: {type(value).__name__}"
        )

    print()
    print("URL")
    print("---")
    print(
        example.get(
            "url",
            "",
        )
    )

    print()
    print("Date")
    print("----")
    print(
        example.get(
            "date",
            "",
        )
    )

    print()
    print("Metadata")
    print("--------")
    print(
        example.get(
            "metadata",
            "",
        )
    )

    print()
    print("Text preview")
    print("------------")

    text = example.get(
        "text",
        "",
    )

    print(
        text[:3000]
    )

    print()
    print(
        f"Full document characters: "
        f"{len(text):,}"
    )


if __name__ == "__main__":
    main()