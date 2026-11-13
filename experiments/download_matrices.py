"""Загрузить явно выбранные матрицы SuiteSparse Matrix Collection.

Пример:
    python -m experiments.download_matrices nos4 bcsstk14

Скрипт не выбирает вручную «удобные» матрицы за пользователя. Имена задаются
явно; группа проверяется по официальному индексу. Для воспроизводимой выборки
сохраните список имён и манифест рядом с результатами.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
INDEX_URL = "https://sparse.tamu.edu/files/ssstats.csv"
ARCHIVE_URL = "https://sparse.tamu.edu/MM/{group}/{name}.tar.gz"
FIELDS = ["group", "name", "nrows", "ncols", "nnz", "isReal", "isBinary", "isND",
          "posdef", "pattern_symmetry", "numerical_symmetry", "kind"]


def get_bytes(url: str, timeout: float = 120.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "slae-vkr-research/2.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def sha256_file(path: str | Path) -> str:
    """Хешировать файл блоками, не загружая крупную матрицу целиком в память."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_index_row(row: list[str]) -> dict | None:
    if len(row) < len(FIELDS):
        return None
    item = dict(zip(FIELDS, [cell.strip() for cell in row[:len(FIELDS)]]))
    try:
        int(item["nrows"])
        int(item["ncols"])
        int(item["nnz"])
    except (TypeError, ValueError):
        return None
    if not item["name"] or not item["group"]:
        return None
    return item


def read_index(data_dir: str | Path | None = None, refresh: bool = False) -> dict[str, list[dict]]:
    """Загрузить и разобрать официальный CSV-индекс в выбранный каталог."""
    data_dir = Path(data_dir) if data_dir is not None else ROOT / "data" / "matrices"
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / "ssstats.csv"
    if refresh or not cache.exists():
        payload = get_bytes(INDEX_URL)
        temporary = cache.with_suffix(".csv.tmp")
        temporary.write_bytes(payload)
        temporary.replace(cache)
    raw = cache.read_text(encoding="utf-8-sig")
    by_name: dict[str, list[dict]] = {}
    for row in csv.reader(io.StringIO(raw)):
        parsed = _valid_index_row(row)
        if parsed is not None:
            by_name.setdefault(parsed["name"], []).append(parsed)
    if not by_name:
        raise RuntimeError("не удалось разобрать официальный ssstats.csv; проверьте формат файла")
    return by_name


def _safe_component(value: str, label: str) -> str:
    value = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in {".", ".."}:
        raise ValueError(f"недопустимое значение {label}: {value!r}")
    return value


def parse_group_overrides(values: list[str]) -> dict[str, str]:
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"группа должна задаваться как ИМЯ=ГРУППА: {value}")
        name, group = value.split("=", 1)
        safe_name = _safe_component(name, "имя матрицы")
        if safe_name in overrides:
            raise ValueError(f"группа для {safe_name} указана повторно")
        overrides[safe_name] = _safe_component(group, "группа")
    return overrides


def _matrix_market_size(payload: bytes, expected_name: str) -> tuple[int, int]:
    """Проверить заголовок Matrix Market и извлечь размеры без загрузки всей матрицы."""
    stream = io.BytesIO(payload)
    header = stream.readline().decode("ascii", errors="replace").strip().lower()
    if not header.startswith("%%matrixmarket matrix"):
        raise ValueError(f"{expected_name}: файл не является Matrix Market")
    if "complex" in header:
        raise ValueError(f"{expected_name}: комплексная матрица не поддерживается")
    if not any(token in header.split() for token in ("pattern", "integer", "real")):
        raise ValueError(f"{expected_name}: неизвестный тип элементов Matrix Market")
    for line in stream:
        text = line.decode("ascii", errors="replace").strip()
        if not text or text.startswith("%"):
            continue
        columns = text.split()
        if len(columns) not in (2, 3):
            raise ValueError(f"{expected_name}: некорректная строка размеров Matrix Market")
        rows, cols = int(columns[0]), int(columns[1])
        if rows < 1 or cols < 1:
            raise ValueError(f"{expected_name}: матрица имеет нулевой размер")
        return rows, cols
    raise ValueError(f"{expected_name}: в файле не найдена строка размеров")


def download_one(name: str, group: str, data_dir: Path, expected: dict | None = None) -> str:
    name = _safe_component(name, "имя матрицы")
    group = _safe_component(group, "группа")
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    target = (data_dir / f"{name}.mtx").resolve()
    if target.parent != data_dir:
        raise ValueError("путь файла матрицы выходит за пределы каталога данных")
    if target.exists():
        payload = target.read_bytes()
        rows, cols = _matrix_market_size(payload, name)
        if rows != cols:
            raise ValueError(f"локальный файл {target} не является квадратной матрицей")
        if expected:
            expected_shape = (int(expected["nrows"]), int(expected["ncols"]))
            if (rows, cols) != expected_shape:
                raise ValueError(f"локальная матрица {(rows, cols)} не совпадает с индексом {expected_shape}")
        return "уже существует (проверены тип и размеры Matrix Market)"

    archive = get_bytes(ARCHIVE_URL.format(group=group, name=name))
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        matches = [member for member in tar.getmembers()
                   if not member.isdir() and PurePosixPath(member.name).name == f"{name}.mtx"]
        if len(matches) != 1:
            raise ValueError(f"в архиве найдено {len(matches)} файлов {name}.mtx, ожидался ровно один")
        member = matches[0]
        if not member.isfile() or member.issym() or member.islnk():
            raise ValueError("небезопасный тип файла матрицы в tar-архиве")
        source = tar.extractfile(member)
        if source is None:
            raise ValueError("не удалось прочитать файл матрицы из архива")
        with source:
            payload = source.read()

    rows, cols = _matrix_market_size(payload, name)
    if rows != cols:
        raise ValueError(f"{name}: матрица не квадратная ({rows} x {cols})")
    if expected:
        expected_shape = (int(expected["nrows"]), int(expected["ncols"]))
        if (rows, cols) != expected_shape:
            raise ValueError(f"{name}: размер файла {(rows, cols)} не совпадает с индексом {expected_shape}")

    temporary = target.with_suffix(".mtx.tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return f"загружено (объём {len(payload) / 2**20:.2f} МиБ, размер матрицы {rows})"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+", help="имена матриц из официальной коллекции")
    parser.add_argument("--data", default=str(ROOT / "data" / "matrices"), help="каталог матриц")
    parser.add_argument("--group", action="append", default=[], metavar="ИМЯ=ГРУППА",
                        help="указать группу, если одно имя матрицы встречается в нескольких группах")
    parser.add_argument("--refresh-index", action="store_true",
                        help="загрузить актуальный список матриц из официального каталога")
    args = parser.parse_args()

    try:
        overrides = parse_group_overrides(args.group)
    except ValueError as exc:
        parser.error(str(exc))
    data_dir = Path(args.data).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        index = read_index(data_dir=data_dir, refresh=args.refresh_index)
    except Exception as exc:
        if not overrides or any(name not in overrides for name in args.names):
            raise SystemExit(f"Не удалось получить официальный индекс: {exc}\n"
                             "При наличии проверенных локальных метаданных задайте --group ИМЯ=ГРУППА.")
        index = {}

    rows = []
    for raw_name in args.names:
        try:
            name = _safe_component(raw_name, "имя матрицы")
        except ValueError as exc:
            print(f"{raw_name}: Ошибка {exc}")
            rows.append({"name": str(raw_name), "status": "failed", "error": str(exc)[:240]})
            continue
        entries = index.get(name, [])
        if name in overrides:
            group = overrides[name]
            meta = next((entry for entry in entries if entry["group"] == group), None)
            if entries and meta is None:
                raise SystemExit(f"группа {group} для {name} отсутствует в официальном индексе")
        else:
            if not entries:
                print(f"{name}: нет записи в официальном индексе; пропущено")
                rows.append({"name": name, "status": "missing_from_index"})
                continue
            # Если имя встречается несколько раз в разных группах, выбор не делается автоматически.
            groups = sorted({entry["group"] for entry in entries})
            if len(groups) != 1:
                print(f"{name}: неоднозначная группа {groups}; задайте --group {name}=ГРУППА")
                rows.append({"name": name, "status": "ambiguous_group"})
                continue
            group = groups[0]
            meta = entries[0]
        try:
            status = download_one(name, group, data_dir, meta)
            print(f"{group}/{name}: {status}")
            rows.append({**(meta or {}), "name": name, "group": group, "status": "ok",
                         "sha256": sha256_file(data_dir / f"{name}.mtx")})
        except Exception as exc:
            print(f"{group}/{name}: Ошибка {type(exc).__name__}: {exc}")
            rows.append({**(meta or {}), "name": name, "group": group,
                         "status": "failed", "error": f"{type(exc).__name__}: {exc}"[:240]})

    manifest = data_dir / "manifest.csv"
    fields = list(dict.fromkeys(FIELDS + ["status", "error", "sha256"]))
    temporary = manifest.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(manifest)
    failed = [row for row in rows if row.get("status") != "ok"]
    if failed:
        raise SystemExit(f"Загрузка завершена с {len(failed)} проблемами. Подробности: {manifest}")


if __name__ == "__main__":
    main()
