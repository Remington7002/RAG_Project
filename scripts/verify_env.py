import sys
import importlib


REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "sentence_transformers",
    "chromadb",
    "spacy",
]


def check_python():
    print("=" * 50)
    print("PYTHON ENVIRONMENT")
    print("=" * 50)

    print(f"Python version: {sys.version}")

    if sys.version_info < (3, 12):
        raise RuntimeError("Python 3.12 or newer is required.")

    print("Python version: OK")


def check_packages():
    print("\n" + "=" * 50)
    print("PACKAGE IMPORTS")
    print("=" * 50)

    for package in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown")
            print(f"{package}: OK (version: {version})")
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import {package}: {e}"
            )


def check_torch():
    print("\n" + "=" * 50)
    print("PYTORCH / GPU")
    print("=" * 50)

    try:
        import torch
    except ImportError:
        raise RuntimeError("PyTorch is not installed.")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("GPU: Not available")
        print("Running with CPU")


def main():
    print("\nRAG PROJECT ENVIRONMENT CHECK\n")

    check_python()
    check_packages()
    check_torch()

    print("\n" + "=" * 50)
    print("ENVIRONMENT CHECK PASSED")
    print("=" * 50)


if __name__ == "__main__":
    main()
