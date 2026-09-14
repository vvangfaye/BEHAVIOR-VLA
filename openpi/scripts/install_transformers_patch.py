"""Install or verify OpenPI's versioned Transformers extensions in this interpreter."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import pathlib
import shutil


def main(*, check: bool = False):
    version = importlib.metadata.version("transformers")
    if version != "5.5.4":
        raise RuntimeError(f"Expected transformers==5.5.4, found {version}. Use scripts/setup_pytorch_env.sh.")
    spec = importlib.util.find_spec("transformers")
    if spec is None or spec.origin is None:
        raise RuntimeError("Transformers is not installed in this interpreter.")
    target = pathlib.Path(spec.origin).parent
    source = pathlib.Path(__file__).resolve().parents[1] / "src/openpi/models_pytorch/transformers_replace"
    files = sorted(source.rglob("*.py"))
    if not files:
        raise RuntimeError(f"No Transformers extensions found in {source}")
    for path in files:
        dest = target / path.relative_to(source)
        expected = hashlib.sha256(path.read_bytes()).digest()
        matches = dest.is_file() and hashlib.sha256(dest.read_bytes()).digest() == expected
        if matches:
            continue
        if check:
            raise RuntimeError(
                f"Missing or outdated OpenPI extension: {dest}. "
                "Run this script without --check using the Torch environment's Python."
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        backup = dest.with_suffix(dest.suffix + ".openpi-original")
        if dest.exists() and not backup.exists():
            shutil.copy2(dest, backup)
        temporary = dest.with_suffix(dest.suffix + ".openpi-tmp")
        shutil.copy2(path, temporary)
        temporary.replace(dest)
    print(f"OpenPI Transformers extensions {'verified' if check else 'installed'}: {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify files without modifying the environment.")
    main(**vars(parser.parse_args()))
