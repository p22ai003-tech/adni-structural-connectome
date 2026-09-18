#!/usr/bin/env python3
"""Correct PptxGenJS presentation.xml child ordering for strict Office readers.

PptxGenJS 4.0.1 emits p:notesMasterIdLst after p:sldIdLst.  The transitional
PresentationML schema requires notesMasterIdLst before sldIdLst.  Some readers
accept the order; desktop PowerPoint can request repair.  This post-processor
changes only that element position and rewrites the package atomically.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import zipfile
from pathlib import Path

from lxml import etree


P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
PRESENTATION_XML = "ppt/presentation.xml"


def corrected_xml(payload: bytes) -> tuple[bytes, bool]:
    root = etree.fromstring(payload)
    notes = root.find(f"{{{P_NS}}}notesMasterIdLst")
    slides = root.find(f"{{{P_NS}}}sldIdLst")
    if notes is None or slides is None:
        return payload, False
    children = list(root)
    if children.index(notes) < children.index(slides):
        return payload, False
    root.remove(notes)
    root.insert(list(root).index(slides), notes)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True), True


def fix_package(path: Path) -> bool:
    with zipfile.ZipFile(path, "r") as source:
        original = source.read(PRESENTATION_XML)
        replacement, changed = corrected_xml(original)
        if not changed:
            return False
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        os.close(fd)
        temp = Path(temp_name)
        try:
            with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as destination:
                for info in source.infolist():
                    data = replacement if info.filename == PRESENTATION_XML else source.read(info.filename)
                    destination.writestr(info, data)
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.paths:
        changed = fix_package(path.resolve())
        print(f"{path.resolve()}\t{'CORRECTED' if changed else 'UNCHANGED'}")


if __name__ == "__main__":
    main()
