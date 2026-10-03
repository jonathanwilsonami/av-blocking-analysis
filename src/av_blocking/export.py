"""Export notebook artifacts so the Quarto paper and slides can reference them.

* :func:`save_fig`   -> ``project-site/assets/figures/<name>.png``
* :func:`save_table` -> ``project-site/assets/tables/<name>.md`` (+ ``.csv``); the
  markdown carries a Quarto caption and ``#tbl-<name>`` label so the paper can
  ``{{< include >}}`` it and cross-reference it with ``@tbl-<name>``.
* :func:`save_text`  -> ``project-site/assets/snippets/<name>.md`` for prose that
  depends on results.
* :func:`save_var`   -> merges a value into ``project-site/_variables.yml`` so the
  paper can use it inline as ``{{< var key >}}``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from . import config


def _ensure(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_fig(fig, name: str, dpi: int = 200, directory: Path = config.FIG_DIR) -> Path:
    path = _ensure(directory) / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path


def save_table(
    df: pd.DataFrame,
    name: str,
    caption: str,
    index: bool = False,
    floatfmt: str = ".3g",
    directory: Path = config.TABLE_DIR,
) -> Path:
    d = _ensure(directory)
    df.to_csv(d / f"{name}.csv", index=index)
    md = df.to_markdown(index=index, floatfmt=floatfmt)
    label = name if name.startswith("tbl-") else f"tbl-{name}"
    path = d / f"{name}.md"
    path.write_text(f"{md}\n\n: {caption} {{#{label}}}\n")
    return path


def save_text(text: str, name: str, directory: Path = config.SNIPPET_DIR) -> Path:
    path = _ensure(directory) / f"{name}.md"
    path.write_text(text.strip() + "\n")
    return path


def _plain(v):
    if hasattr(v, "item"):
        v = v.item()
    # Floats are written as strings: Quarto would otherwise render e.g. 0.064 as 6.4e-2.
    if isinstance(v, float):
        return format(v, "g")
    return v


def save_var(key: str, value, path: Path = config.VARIABLES_FILE) -> None:
    """Set ``key`` (dotted for nesting, e.g. ``"freq.total"``) in ``_variables.yml``."""
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    data = data or {}
    node = data
    *parents, leaf = key.split(".")
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = _plain(value)
    path.write_text(yaml.safe_dump(data, sort_keys=True, allow_unicode=True))


def save_vars(values: dict, prefix: str = "") -> None:
    for k, v in values.items():
        save_var(f"{prefix}.{k}" if prefix else k, v)
