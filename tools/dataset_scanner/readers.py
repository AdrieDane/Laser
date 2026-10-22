"""Dependency-free, bounded readers for XLSX and numeric MATLAB v5 structs.

Never execute workbook formulas, macros, MATLAB code, pickle or R objects.
MATLAB v7.3/HDF5 and unfamiliar data types are explicitly unsupported, not passes.
"""

from __future__ import annotations

from array import array
import csv
import io
import math
from pathlib import Path, PurePosixPath
import struct
import sys
import xml.etree.ElementTree as ET
import zipfile
import zlib

MAX_XML = 256 * 1024**2
MAX_MAT = 1024**3
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


class Workbook:
    def __init__(self, path):
        self.zip = zipfile.ZipFile(path)
        self.strings = []
        if "xl/sharedStrings.xml" in self.zip.namelist():
            root = ET.fromstring(self._read("xl/sharedStrings.xml"))
            self.strings = ["".join(x.itertext()) for x in root.findall(NS + "si")]
        root = ET.fromstring(self._read("xl/workbook.xml"))
        rels = ET.fromstring(self._read("xl/_rels/workbook.xml.rels"))
        targets = {}
        for rel in rels:
            target = rel.get("Target", "")
            if rel.get("TargetMode") == "External":
                continue
            full = target.lstrip("/") if target.startswith("/") else "xl/" + target
            if ".." in PurePosixPath(full).parts:
                raise ValueError("Unsafe workbook relationship")
            targets[rel.get("Id")] = full
        self.sheets = {}
        for sheet in root.findall(NS + "sheets/" + NS + "sheet"):
            key = sheet.get(
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            )
            if key in targets:
                self.sheets[sheet.get("name")] = targets[key]

    def _read(self, name):
        if self.zip.getinfo(name).file_size > MAX_XML:
            raise ValueError("Workbook component too large for safe inspection")
        return self.zip.read(name)

    def rows(self, sheet):
        target = self.sheets[sheet]
        if self.zip.getinfo(target).file_size > MAX_XML:
            raise ValueError("Worksheet exceeds safe inspection limit")
        with self.zip.open(target) as source:
            for _, row in ET.iterparse(source, events=("end",)):
                if row.tag != NS + "row":
                    continue
                values = []
                for cell in row.findall(NS + "c"):
                    coordinate = cell.get("r", "")
                    col = 0
                    for char in coordinate:
                        if not char.isalpha():
                            break
                        col = 26 * col + ord(char.upper()) - 64
                    if not 1 <= col <= 16384:
                        raise ValueError("Invalid worksheet coordinate")
                    while len(values) < col:
                        values.append("")
                    kind = cell.get("t", "")
                    value = cell.findtext(NS + "v", "")
                    if kind == "s":
                        value = self.strings[int(value)] if value else ""
                    elif kind == "inlineStr":
                        inline = cell.find(NS + "is")
                        value = "".join(inline.itertext()) if inline is not None else ""
                    elif kind == "e":
                        value = ""  # Unavailable/error values never become zero.
                    values[col - 1] = value
                yield values
                row.clear()

    def records(self, sheet):
        rows = iter(self.rows(sheet))
        header = [str(v).strip() for v in next(rows)]
        for row in rows:
            yield dict(zip(header, row + [""] * max(0, len(header) - len(row))))

    def close(self):
        self.zip.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def table_records(path):
    if Path(path).suffix.lower() == ".xlsx":
        with Workbook(path) as book:
            yield from book.records(next(iter(book.sheets)))
    else:
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(4096)
            handle.seek(0)
            delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
            yield from csv.DictReader(handle, delimiter=delimiter)


class Numeric:
    def __init__(self, shape, values):
        self.shape, self.values = shape, values


class MatReader:
    """Selective v5 struct reader: skip unneeded payloads, cap decompression."""

    wanted = {
        "MS_data",
        "RP",
        "NP",
        "mz_grouped",
        "RT_grouped",
        "RTp_grouped",
        "annotation",
    }

    def __init__(self, endian):
        self.endian = endian

    def tag(self, source):
        data = source.read(8)
        if not data:
            return None
        if len(data) != 8:
            raise ValueError("Truncated MAT tag")
        small_kind, small_size = struct.unpack(self.endian + "HH", data[:4])
        if 0 < small_size <= 4:
            return small_kind, small_size, data[4 : 4 + small_size], source.tell()
        kind, size = struct.unpack(self.endian + "II", data)
        if size > MAX_MAT:
            raise ValueError("MAT payload exceeds safety limit")
        position = source.tell()
        # MATLAB miCOMPRESSED elements need not be padded to eight bytes.
        end = position + size if kind == 15 else position + ((size + 7) // 8) * 8
        return kind, size, None, end

    def payload(self, source, tag):
        _, size, small, end = tag
        data = small if small is not None else source.read(size)
        if len(data) != size:
            raise ValueError("Truncated MAT payload")
        source.seek(end)
        return data

    def matrix(self, source, tag, force=False):
        if tag[0] != 14:
            raise ValueError("Expected MAT matrix")
        flags = self.payload(source, self.tag(source))
        cls = struct.unpack(self.endian + "I", flags[:4])[0]
        if cls & 0x800:
            raise ValueError("Complex MAT matrices are unsupported")
        dims = self.payload(source, self.tag(source))
        shape = struct.unpack(self.endian + "i" * (len(dims) // 4), dims)
        name = self.payload(source, self.tag(source)).decode("utf-8", errors="strict")
        if not force and name not in self.wanted:
            source.seek(tag[3])
            return name, None
        count = math.prod(shape)
        if cls & 0xFF == 2:
            if count != 1:
                raise ValueError("Only scalar MAT structs are supported")
            length = self.payload(source, self.tag(source))
            width = struct.unpack(self.endian + "i", length[:4])[0]
            names = self.payload(source, self.tag(source))
            if width <= 0 or len(names) % width:
                raise ValueError("Invalid MAT struct names")
            fields = [
                names[i : i + width].split(b"\0", 1)[0].decode()
                for i in range(0, len(names), width)
            ]
            value = {}
            for field in fields:
                child = self.tag(source)
                if child is None:
                    raise ValueError("Missing MAT struct field")
                if field in self.wanted:
                    _, value[field] = self.matrix(source, child, force=True)
                else:
                    source.seek(child[3])
        else:
            numeric = self.tag(source)
            code = {
                1: "b",
                2: "B",
                3: "h",
                4: "H",
                5: "i",
                6: "I",
                7: "f",
                9: "d",
                12: "q",
                13: "Q",
            }.get(numeric[0])
            if code is None:
                raise ValueError("Unsupported MAT numeric storage type")
            values = array(code)
            values.frombytes(self.payload(source, numeric))
            if (self.endian == "<") != (sys.byteorder == "little"):
                values.byteswap()
            if len(values) != count:
                raise ValueError("MAT matrix dimensions do not match payload")
            value = Numeric(shape, values)
        source.seek(tag[3])
        return name, value

    def elements(self, source):
        result = {}
        while (tag := self.tag(source)) is not None:
            if tag[0] == 15:
                compressed = self.payload(source, tag)
                decoder = zlib.decompressobj()
                raw = decoder.decompress(compressed, MAX_MAT + 1)
                if len(raw) > MAX_MAT or not decoder.eof:
                    raise ValueError("MAT decompression limit or truncated data")
                result.update(self.elements(io.BytesIO(raw)))
            elif tag[0] == 14:
                name, value = self.matrix(source, tag)
                if value is not None:
                    result[name] = value
            else:
                source.seek(tag[3])
        return result


def load_rt_predictions(path):
    with Path(path).open("rb") as handle:
        header = handle.read(128)
        if (
            len(header) != 128
            or header[:6] != b"MATLAB"
            or header.startswith(b"MATLAB 7.3")
            or header[126:128] not in (b"IM", b"MI")
        ):
            raise ValueError("Need MATLAB v5/v7 MAT file; v7.3/HDF5 is not supported")
        reader = MatReader("<" if header[126:128] == b"IM" else ">")
        data = reader.elements(handle).get("MS_data", {})
    output = {}
    for phase in ("RP", "NP"):
        fields = data.get(phase, {})
        if (
            not {"mz_grouped", "RT_grouped", "RTp_grouped", "annotation"}
            <= fields.keys()
        ):
            raise ValueError(f"Missing saved RT/annotation fields for {phase}")
        mz, rt, pred, ann = (
            fields[k] for k in ("mz_grouped", "RT_grouped", "RTp_grouped", "annotation")
        )
        count = len(mz.values)
        if (
            len(rt.values) != count
            or len(pred.values) != count
            or ann.shape != (count, 2)
        ):
            raise ValueError(f"Inconsistent saved {phase} dimensions")
        output[phase] = [
            (mz.values[i], rt.values[i], pred.values[i])
            for i in range(count)
            if ann.values[i] > 0
        ]
    return output
