import argparse

from src.data.synthetic import generate_demo_dataset
from src.data.cache import load_or_build_split
from src.data.splits import get_split, map_symbol_to_aami_class
from src.utils.config import load_config
from src.data.mitbih import RAW_DIR
from src.utils.paths import DATA_DIR


def download_mitbih() -> None:
    import wfdb

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database("mitdb", str(RAW_DIR))
    print(f"MIT-BIH records saved to {RAW_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch or generate ECG Research Project datasets")
    parser.add_argument("--mode", choices=["research", "demo"], default="demo")
    parser.add_argument("--task", choices=["binary", "aami", "both"], default="binary")
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()

    if args.mode == "research":
        download_mitbih()
    else:
        generate_demo_dataset()
        print(f"Synthetic demo fixture written to {DATA_DIR / 'demo'}")

    cfg = load_config("dataset.yaml")
    for name, records in get_split(args.mode, cfg).items():
        tasks = ["binary", "aami"] if args.task == "both" else [args.task]
        for task in tasks:
            X, y, _ = load_or_build_split(
                args.mode, name, records, cfg,
                label_fn=map_symbol_to_aami_class if task == "aami" else None,
                cache_suffix="aami" if task == "aami" else "",
                force_rebuild=args.force_rebuild,
            )
            print(f"Prepared {args.mode} {task} {name}: {len(y):,} beats")


if __name__ == "__main__":
    main()
