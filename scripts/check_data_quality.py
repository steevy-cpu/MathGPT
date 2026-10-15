import json
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_data_quality(file_path):
    total_documents = 0

    empty_documents = 0
    short_documents = 0
    duplicate_documents = 0

    seen_texts = set()

    domains = {}

    quality_buckets = {
        "below_3": 0,
        "3_to_4": 0,
        "4_to_5": 0,
    }

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:
            document = json.loads(line)

            total_documents += 1

            text = document.get("text", "").strip()
            url = document.get("url", "")
            score = document.get("quality_score")

            # Check empty text
            if not text:
                empty_documents += 1

            # Check very short text
            if len(text) < 200:
                short_documents += 1

            # Check exact duplicates
            if text in seen_texts:
                duplicate_documents += 1
            else:
                seen_texts.add(text)

            # Count domains
            if url:
                domain = urlparse(url).netloc

                if domain:
                    domains[domain] = domains.get(domain, 0) + 1

            # Quality score buckets
            if score is not None:

                if score < 3:
                    quality_buckets["below_3"] += 1

                elif score < 4:
                    quality_buckets["3_to_4"] += 1

                else:
                    quality_buckets["4_to_5"] += 1

    print()
    print("Data Quality Report")
    print("-------------------")

    print(f"Documents: {total_documents:,}")
    print(f"Empty documents: {empty_documents:,}")
    print(f"Short documents (<200 chars): {short_documents:,}")
    print(f"Exact duplicates: {duplicate_documents:,}")

    print()
    print("Quality Score Distribution")
    print("--------------------------")

    print(f"Below 3: {quality_buckets['below_3']}")
    print(f"3 to <4: {quality_buckets['3_to_4']}")
    print(f"4 to 5: {quality_buckets['4_to_5']}")

    print()
    print("Most Common Domains")
    print("-------------------")

    sorted_domains = sorted(
        domains.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for domain, count in sorted_domains[:10]:
        print(f"{domain}: {count}")

    print()


def main():

    file_path = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "general"
        / "fineweb_edu_sample.jsonl"
    )

    check_data_quality(file_path)


if __name__ == "__main__":
    main()