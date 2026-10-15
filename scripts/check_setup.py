import json
from pathlib import Path


# Find the root folder of the project.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_json(file_path):
    """
    Load a JSON file and return its contents as a Python dictionary.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def check_poc_config(config):
    """
    Perform basic checks on the main POC configuration.
    """

    required_sections = [
        "project_name",
        "version",
        "model",
        "tokenizer",
        "general_pretraining",
        "math_pretraining",
        "training",
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing section in POC config: {section}")

    print("✓ POC configuration looks valid")


def check_manifest(manifest):
    """
    Perform basic checks on a dataset manifest.
    """

    required_fields = [
        "dataset_name",
        "version",
        "target_tokens",
        "sources",
    ]

    for field in required_fields:
        if field not in manifest:
            raise ValueError(
                f"Missing field in dataset manifest: {field}"
            )

    # If the manifest has data sources,
    # verify that their percentages add up to 100.
    if manifest["sources"]:

        total_percentage = sum(
            source["target_percentage"]
            for source in manifest["sources"]
        )

        if total_percentage != 100:
            raise ValueError(
                f"Source percentages add up to "
                f"{total_percentage}, not 100"
            )

        total_source_tokens = sum(
            source["target_tokens"]
            for source in manifest["sources"]
        )

        if total_source_tokens != manifest["target_tokens"]:
            raise ValueError(
                "Source token targets do not match "
                "the manifest target_tokens value"
            )

    print(
        f"✓ {manifest['dataset_name']} "
        f"v{manifest['version']} looks valid"
    )


def main():

    config_path = PROJECT_ROOT / "config" / "poc_config.json"

    general_manifest_path = (
        PROJECT_ROOT
        / "data"
        / "manifests"
        / "general_v0.1.json"
    )

    math_manifest_path = (
        PROJECT_ROOT
        / "data"
        / "manifests"
        / "math_v0.1.json"
    )

    print("\nChecking MathGPT POC setup...\n")

    config = load_json(config_path)
    general_manifest = load_json(general_manifest_path)
    math_manifest = load_json(math_manifest_path)

    check_poc_config(config)

    check_manifest(general_manifest)
    check_manifest(math_manifest)

    print("\n✓ All configuration files passed validation.\n")


if __name__ == "__main__":
    main()