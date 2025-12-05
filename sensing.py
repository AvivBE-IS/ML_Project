from pathlib import Path
import csv
# Removed tqdm import as it is not installed in the environment

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUTPUT_FILE = ROOT / "sensed_data.csv"
LABELS = ["culture", "news", "opinion", "sport"]


def parse_article(file_path, label):
    """
    Reads a .txt file and extracts its components based on the known structure:
    Line 1: The Guardian | Section | Date
    Line 2: By Author
    Line 3: URL
    Line 4: Title
    Line 5: Separator (------)
    Line 6+: Body
    """
    try:
        text = file_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        # Initialize variables in case parsing fails
        date = ""
        author = ""
        url = ""
        title = ""
        body = ""

        # Parsing logic based on your save_article format
        if len(lines) > 5:
            # Line 1 example: "The Guardian | sport | 2025-12-05"
            header_parts = lines[0].split("|")
            if len(header_parts) >= 3:
                date = header_parts[-1].strip()

            # Line 2 example: "By John Doe"
            if lines[1].startswith("By "):
                author = lines[1][3:].strip()
            else:
                author = lines[1].strip()

            # Line 3: URL
            url = lines[2].strip()

            # Line 4 is empty (due to double newline in data_collection.py)
            # Line 5 contains the Title
            title = lines[4].strip()

            # Body starts after the separator line (Line 5 is usually empty or dashes)
            # We look for the separator line "------"
            body_start_index = 5
            for i, line in enumerate(lines):
                if set(line.strip()) == {"-"}:  # Finds the separator line
                    body_start_index = i + 1
                    break

            body = "\n".join(lines[body_start_index:]).strip()

        else:
            # Fallback for unstructured files
            body = text

        return {
            "filename": file_path.name,
            "label": label,
            "date": date,
            "author": author,
            "title": title,
            "url": url,
            "body": body
        }

    except Exception as e:
        print(f"Error parsing {file_path.name}: {e}")
        return None


def main():
    all_rows = []

    print(f"Starting Sensing process...")

    for label in LABELS:
        folder = DATA_DIR / label
        if not folder.exists():
            print(f"Warning: Folder {label} not found.")
            continue

        files = list(folder.glob("*.txt"))
        print(f"Processing {label} ({len(files)} files)...")

        # Removed tqdm wrapper to avoid missing dependency error
        for file_path in files:
            row = parse_article(file_path, label)
            if row:
                all_rows.append(row)

    if all_rows:
        keys = all_rows[0].keys()
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"\nSuccess! Sensed data saved to: {OUTPUT_FILE}")
        print(f"Total records: {len(all_rows)}")
    else:
        print("No data found.")


if __name__ == "__main__":
    main()