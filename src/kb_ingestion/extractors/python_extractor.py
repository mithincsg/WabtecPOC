from __future__ import annotations

import ast
from pathlib import Path

from . import ExtractedUnit

_DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_METHOD_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)


def _node_source(lines: list[str], node: ast.AST) -> str:
    """Slice full source text for a def/class node, including any decorators
    (ast.get_source_segment excludes the '@...' lines above a decorated def).
    """
    start_line = node.lineno
    for decorator in getattr(node, "decorator_list", []):
        start_line = min(start_line, decorator.lineno)
    end_line = node.end_lineno
    return "\n".join(lines[start_line - 1 : end_line])


def _is_docstring_stmt(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _class_header_source(lines: list[str], node: ast.ClassDef) -> str:
    """Just the `class Foo(Base):` signature + docstring, not the full body -
    used when a class's methods are chunked separately, so retrieving "what
    is this class for" doesn't require pulling in one arbitrary method.
    """
    start_line = node.lineno
    for decorator in getattr(node, "decorator_list", []):
        start_line = min(start_line, decorator.lineno)

    end_line = node.lineno
    if node.body:
        first = node.body[0]
        end_line = max(start_line, first.lineno - 1)
        if _is_docstring_stmt(first):
            end_line = first.end_lineno

    return "\n".join(lines[start_line - 1 : end_line])


class PythonExtractor:
    """Module -> class -> method/function structure, mirroring how Python
    code is actually organized so chunks never split a method:

    - A module docstring becomes its own unit.
    - A class with methods becomes a compact class-header unit (signature +
      docstring only) plus one unit per method - a method is the natural
      retrieval granularity for a "Python API" and stays whole.
    - A class with no methods (attribute-only classes, Enums) stays a single
      unit, since there's no meaningful sub-structure to split on.
    - Module-level functions are unchanged: one unit each.

    Falls back to whole-file text if the file has a syntax error.
    """

    def extract(self, file_path: Path) -> list[ExtractedUnit]:
        source = file_path.read_text(encoding="utf-8", errors="replace")

        try:
            tree = ast.parse(source, filename=str(file_path))
        except SyntaxError:
            return [
                ExtractedUnit(
                    text=source,
                    unit_type="raw_python",
                    locator={"reason": "syntax_error"},
                )
            ]

        units: list[ExtractedUnit] = []
        lines = source.splitlines()

        module_docstring = ast.get_docstring(tree)
        if module_docstring:
            units.append(
                ExtractedUnit(
                    text=module_docstring,
                    unit_type="module_docstring",
                    locator={"lineno": 1},
                )
            )

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                units.extend(self._extract_class(lines, node))
            elif isinstance(node, _METHOD_TYPES):
                units.extend(self._extract_function(lines, node))

        if not units:
            # No module docstring and no top-level defs/classes (e.g. a
            # script with only top-level statements) - keep the whole file.
            units.append(
                ExtractedUnit(
                    text=source,
                    unit_type="raw_python",
                    locator={"reason": "no_top_level_defs"},
                )
            )

        return units

    def _extract_class(self, lines: list[str], node: ast.ClassDef) -> list[ExtractedUnit]:
        methods = [n for n in node.body if isinstance(n, _METHOD_TYPES)]
        if not methods:
            text = _node_source(lines, node)
            if not text.strip():
                return []
            return [
                ExtractedUnit(
                    text=text,
                    unit_type="class",
                    locator={"name": node.name, "lineno": node.lineno},
                    extra={"docstring": ast.get_docstring(node) or ""},
                )
            ]

        units: list[ExtractedUnit] = []
        header_text = _class_header_source(lines, node)
        if header_text.strip():
            units.append(
                ExtractedUnit(
                    text=header_text,
                    unit_type="class",
                    locator={"name": node.name, "lineno": node.lineno},
                    extra={"docstring": ast.get_docstring(node) or ""},
                )
            )

        for method in methods:
            method_text = _node_source(lines, method)
            if not method_text.strip():
                continue
            units.append(
                ExtractedUnit(
                    text=method_text,
                    unit_type="function",
                    locator={
                        "name": method.name,
                        "lineno": method.lineno,
                        "class_name": node.name,
                    },
                    extra={"docstring": ast.get_docstring(method) or ""},
                )
            )
        return units

    def _extract_function(
        self, lines: list[str], node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> list[ExtractedUnit]:
        text = _node_source(lines, node)
        if not text.strip():
            return []
        return [
            ExtractedUnit(
                text=text,
                unit_type="function",
                locator={"name": node.name, "lineno": node.lineno},
                extra={"docstring": ast.get_docstring(node) or ""},
            )
        ]
