# tests/test_utils.py
import os
import sys

_selected_pdfs = None

def get_selected_pdfs():
    global _selected_pdfs

    if _selected_pdfs is not None:
        return _selected_pdfs

    pdf_dir = "sample_pdfs"
    if not os.path.isdir(pdf_dir):
        print(f"❌ Directory '{pdf_dir}' does not exist.")
        sys.exit(1)

    all_pdfs = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]

    if not all_pdfs:
        print("❌ No PDFs found, so can't run the tests.")
        sys.exit(1)

    print("\n📂 Available PDFs:")
    for i, f in enumerate(all_pdfs, start=1):
        print(f"  [{i}] {f}")

    try:
        count = int(input("\nEnter number of PDFs to use for tests: "))
        _selected_pdfs = [os.path.join(pdf_dir, f) for f in all_pdfs[:count]]
        return _selected_pdfs
    except (ValueError, IndexError):
        print("❌ Invalid input.")
        sys.exit(1)
