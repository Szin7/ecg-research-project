import argparse
import yaml
from src.data.protocol import prepare_protocol
from src.data.mitbih import RAW_DIR
from src.utils.paths import CONFIGS_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load((CONFIGS_DIR / "protocol_v2.yaml").read_text())
    if args.download:
        import wfdb

        RAW_DIR.mkdir(parents=True, exist_ok=True)
        records = sorted(
            {str(r) for records in cfg["records"].values() for r in records}
        )
        wfdb.dl_database("mitdb", str(RAW_DIR), records=records)
    prepare_protocol(cfg)
