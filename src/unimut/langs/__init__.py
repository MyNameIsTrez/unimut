"""unimut's per-language backend protocol, shared types, and discovery.

A backend is a plain module exposing:

    EXTENSIONS: set[str]                          -- recognized file extensions
    generate_mutants(file_path, source, *, whole_file=False,
                      changed_lines=None, keep_calls=None) -> list[Mutant]

``Mutant`` and ``MutationError`` (below) are the two shared types every
backend should build its results out of: construct ``unimut.langs.Mutant``
instances directly (don't define your own), and raise
``unimut.langs.MutationError`` (not some backend-specific exception type)
for a region that can't be turned into mutants. That's what lets
unimut's CLI catch a failure from *any* backend -- built-in or a user's
own -- with a single ``except`` clause, without needing to know which
backend raised it.

``whole_file``/``changed_lines``/``keep_calls`` are all optional: a
backend that only supports marker-based regions can omit them entirely,
and unimut refuses ``--diff``/``--whole-file``/``--keep-call`` for that
backend with a clear error rather than silently ignoring the flag (see
``mutate_python.py``, which does exactly this).

Backends are discovered from two places, in this order (later entries
win on a name clash, so a directory backend can deliberately shadow a
built-in one):

1. Every ``mutate_*.py`` file shipped directly inside this package
   (``mutate_c.py``, ``mutate_python.py``, ...). Adding a new built-in
   backend is *only* this: drop the file here. Nothing outside this
   package -- not ``unimut.py``, not any registry dict -- needs editing.

2. Every ``mutate_*.py`` file found in any directory listed in the
   ``UNIMUT_LANG_PATH`` environment variable (``os.pathsep``-separated,
   the same convention as ``$PATH``). This is how a user registers
   *their own* backend without touching unimut's source or packaging at
   all: write a file implementing the protocol above, name it
   ``mutate_<lang>.py``, put it anywhere, and

       export UNIMUT_LANG_PATH=/path/to/that/directory

   after which ``--lang <lang>`` (or a matching entry in that module's
   own ``EXTENSIONS``) picks it up automatically -- including for
   ``--jobs > 1``: the module is imported once, in the main process,
   before any worker is forked, so every worker inherits it already
   loaded.

In both cases, the filename determines the ``--lang`` key: strip the
``mutate_`` prefix and ``.py`` suffix, so ``mutate_rust.py`` registers
as ``rust``. A file that doesn't start with ``mutate_`` is ignored.

A broken *built-in* module (missing ``EXTENSIONS``/``generate_mutants``,
or a genuine import error) is a bug in unimut itself. A broken file
found via ``UNIMUT_LANG_PATH`` is a bug in a user's own plugin. Both are
treated the same way, and both are fatal: :func:`discover_languages`
raises ``MutationError`` either way, and unimut's CLI reports it with a
clean ``unimut: error: ...`` message and a non-zero exit rather than
silently running with an incomplete set of backends. There is no
"skip the bad one and carry on" mode -- if ``UNIMUT_LANG_PATH`` points
at a directory containing a broken ``mutate_*.py``, *every* unimut
invocation fails until it's fixed or removed, not just ones that would
have used it.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Callable, Dict, Optional

_REQUIRED_ATTRS = ("EXTENSIONS", "generate_mutants")

_LANG_PATH_ENV = "UNIMUT_LANG_PATH"


class MutationError(Exception):
    """Raised by a backend when a region cannot be turned into mutants.

    Backends should raise this directly (``from unimut.langs import
    MutationError``) rather than defining their own exception type --
    see the module docstring for why.
    """


@dataclasses.dataclass
class Mutant:
    """One candidate mutation, as produced by any backend's ``generate_mutants``.

    ``mutated`` is ``None`` for mutation kinds with no single replacement
    line to show (pure statement removal, for instance); kinds that swap
    one piece of code for another (an operator flip, a boundary tweak)
    populate it, and unimut prints it as the report's ``+`` line.

    ``_apply`` must be picklable: ``--jobs`` ships ``Mutant`` instances
    to worker processes. A plain closure won't survive that roundtrip --
    define a module-level callable class instead (see
    ``_RemoveStatementApply`` in either backend for the pattern).
    """

    file: str
    line: int
    original: str
    mutated: Optional[str]
    _apply: Callable[[str], str] = dataclasses.field(repr=False, compare=False)

    def apply(self, source: str) -> str:
        """Return a full copy of ``source`` with this mutation applied."""
        return self._apply(source)


def _lang_name(filename: str) -> Optional[str]:
    """``"mutate_rust.py"`` -> ``"rust"``; anything not matching that
    shape -> ``None``."""
    if not filename.startswith("mutate_") or not filename.endswith(".py"):
        return None
    name = filename[len("mutate_") : -len(".py")]
    return name or None


def _validate(module: ModuleType, source: str) -> None:
    missing = [a for a in _REQUIRED_ATTRS if not hasattr(module, a)]
    if missing:
        raise MutationError(
            f"{source} does not implement unimut's backend protocol "
            f"(missing {', '.join(missing)})"
        )


def _builtin_backends() -> Dict[str, ModuleType]:
    """Every ``mutate_*.py`` module shipped inside this package."""
    backends: Dict[str, ModuleType] = {}
    package_dir = Path(__file__).resolve().parent
    for entry in sorted(package_dir.glob("mutate_*.py")):
        lang = _lang_name(entry.name)
        if lang is None:
            continue
        module = importlib.import_module(f"{__name__}.{entry.stem}")
        _validate(module, f"{__name__}.{entry.stem}")
        backends[lang] = module
    return backends


def _load_from_path(path: Path) -> ModuleType:
    """Import ``path`` as its own module, under a stable, unimut-owned name.

    The stable name matters for two reasons: it's what lets a pickled
    ``Mutant`` from this module resolve correctly (pickle looks a
    function up by ``__module__`` + ``__qualname__`` on the receiving
    end), and -- since ``--jobs`` workers are spawned with
    ``multiprocessing``'s ``fork`` context -- a worker inherits the
    parent's already-imported ``sys.modules`` wholesale at fork time, so
    as long as this import happens during startup (before any worker
    exists), no worker ever needs to re-import it itself.
    """
    lang = _lang_name(path.name) or path.stem
    module_name = f"unimut._external_langs.{lang}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise MutationError(f"could not load {path} as a Python module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _path_backends() -> Dict[str, ModuleType]:
    """Extra ``mutate_*.py`` files from ``$UNIMUT_LANG_PATH``.

    Any problem with any one entry here (not a directory, fails to
    import, doesn't implement the protocol) raises ``MutationError``
    immediately -- see the module docstring for why this is fatal
    rather than a skip-and-warn.
    """
    backends: Dict[str, ModuleType] = {}
    raw = os.environ.get(_LANG_PATH_ENV, "")
    for dir_str in raw.split(os.pathsep):
        dir_str = dir_str.strip()
        if not dir_str:
            continue
        directory = Path(dir_str).expanduser()
        if not directory.is_dir():
            raise MutationError(
                f"{_LANG_PATH_ENV} entry '{dir_str}' is not a directory"
            )
        for path in sorted(directory.glob("mutate_*.py")):
            lang = _lang_name(path.name)
            if lang is None:
                continue
            try:
                module = _load_from_path(path)
                _validate(module, str(path))
            except MutationError:
                raise
            except Exception as exc:
                raise MutationError(f"could not load backend '{path}': {exc}") from exc
            backends[lang] = module
    return backends


def discover_languages() -> Dict[str, ModuleType]:
    """All available backends, keyed by ``--lang`` value.

    Raises ``MutationError`` if any backend -- built-in or found via
    ``UNIMUT_LANG_PATH`` -- fails to load or doesn't implement the
    protocol; see the module docstring for why this is fatal rather
    than best-effort.
    """
    backends = _builtin_backends()
    backends.update(_path_backends())
    return backends
