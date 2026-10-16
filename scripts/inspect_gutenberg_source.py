from datasets import load_dataset


def main():
    dataset = load_dataset(
        "common-pile/project_gutenberg",
        split="train",
        streaming=True,
    )

    example = next(iter(dataset))

    print("\nAvailable fields:")
    print("-----------------")

    for key in example.keys():
        print(key)

    print("\nFirst record:")
    print("-------------")

    for key, value in example.items():
        print(f"\n{key}:")
        print(value)


if __name__ == "__main__":
    main()