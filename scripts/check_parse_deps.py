from __future__ import annotations

import importlib.util

mods = {
    "pypdf": "pypdf",
    "openpyxl": "openpyxl",
    "python-docx": "docx",
    "python-pptx": "pptx",
    "unstructured": "unstructured",
    "Pillow": "PIL",
    "httpx": "httpx",
    "openai": "openai",
    "xlrd": "xlrd",
    "beautifulsoup4": "bs4",
    "lxml": "lxml",
    "markdown-it-py": "markdown_it",
}

for label, module_name in mods.items():
    status = "OK" if importlib.util.find_spec(module_name) else "MISSING"
    print(f"{label}: {status}")
