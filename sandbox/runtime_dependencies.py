import gc
import importlib
import importlib.metadata as importlib_metadata
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
from typing import Iterable
from types import ModuleType

try:
    from packaging.requirements import InvalidRequirement, Requirement
except Exception:
    Requirement = None
    InvalidRequirement = Exception


class RuntimeDependencyManager:
    _PREFERRED_OPENCV_PACKAGE = "opencv-contrib-python"
    _OPENCV_PACKAGE_NAMES = {
        "opencv-python",
        "opencv-contrib-python",
        "opencv-python-headless",
        "opencv-contrib-python-headless",
    }
    _BLOCKED_OPENCV_PACKAGES = (
        "opencv-python",
        "opencv-python-headless",
        "opencv-contrib-python-headless",
    )
    _PKG_IMPORT_ALIASES = {
        "pillow": "PIL",
        "opencv-python": "cv2",
        "opencv-contrib-python": "cv2",
        "opencv-python-headless": "cv2",
        "opencv-contrib-python-headless": "cv2",
        "scikit-image": "skimage",
        "pyyaml": "yaml",
        "huggingface-hub": "huggingface_hub",
        "python-dateutil": "dateutil",
        "beautifulsoup4": "bs4",
        "faiss-cpu": "faiss",
        "faiss-gpu": "faiss",
        "pytorch-lightning": "pytorch_lightning",
        "sentence-transformers": "sentence_transformers",
    }
    _MODULE_PACKAGE_ALIASES = {
        "cv2": _PREFERRED_OPENCV_PACKAGE,
        "pil": "pillow",
        "skimage": "scikit-image",
        "yaml": "pyyaml",
        "huggingface_hub": "huggingface-hub",
        "dateutil": "python-dateutil",
        "bs4": "beautifulsoup4",
        "pytorch_lightning": "pytorch-lightning",
        "sentence_transformers": "sentence-transformers",
    }
    _AUTO_INSTALL_MODULE_PACKAGE_ALIASES = {
        "addict": "addict",
        "yaml": "pyyaml",
        "pil": "pillow",
        "cv2": _PREFERRED_OPENCV_PACKAGE,
        "skimage": "scikit-image",
        "dateutil": "python-dateutil",
        "bs4": "beautifulsoup4",
        "huggingface_hub": "huggingface-hub",
        "pytorch_lightning": "pytorch-lightning",
        "sentence_transformers": "sentence-transformers",
    }
    _DIST_COMPAT_GROUPS = {
        "opencv-python": (
            "opencv-contrib-python",
            "opencv-python",
            "opencv-python-headless",
            "opencv-contrib-python-headless",
        ),
        "opencv-contrib-python": (
            "opencv-contrib-python",
        ),
        "opencv-python-headless": (
            "opencv-contrib-python",
            "opencv-python-headless",
            "opencv-contrib-python-headless",
            "opencv-python",
        ),
        "opencv-contrib-python-headless": (
            "opencv-contrib-python",
            "opencv-contrib-python-headless",
            "opencv-python-headless",
            "opencv-python",
        ),
    }
    _CORE_OPTIONAL_MODULES = {
        "cv2": "cv2",
        "skimage": "skimage",
        "PIL": "PIL",
        "torch": "torch",
        "torchvision": "torchvision",
        "modelscope": "modelscope",
        "transformers": "transformers",
        "diffusers": "diffusers",
    }
    _installed_packages: set[str] = set()
    _installed_dist_versions: dict[str, str] = {}

    def __init__(self, base_namespace: dict):
        self._base_namespace = base_namespace
        self._auto_install_attempted: set[str] = set()
        self._auto_install_failed: set[str] = set()

    @staticmethod
    def _canonical_dist_name(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name.strip().lower())

    @staticmethod
    def _is_identifier(name: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))

    @classmethod
    def _extract_dist_name(cls, requirement: str) -> str:
        req = str(requirement or "").strip()
        if not req:
            return ""

        if Requirement is not None:
            try:
                return Requirement(req).name.strip()
            except InvalidRequirement:
                pass
            except Exception:
                pass

        req = req.split(";", maxsplit=1)[0].strip()
        req = req.split("[", maxsplit=1)[0].strip()
        req = re.split(r"(==|!=|>=|<=|>|<|~=)", req, maxsplit=1)[0].strip()
        return req

    @classmethod
    def _replace_requirement_dist_name(cls, requirement: str, replacement: str) -> str:
        dist_name = cls._extract_dist_name(requirement)
        if not dist_name:
            return requirement
        return re.sub(
            rf"^\s*{re.escape(dist_name)}",
            replacement,
            str(requirement),
            count=1,
            flags=re.IGNORECASE
        ).strip()

    @classmethod
    def _normalize_package_name(cls, package: str) -> str:
        normalized = str(package).strip()
        if not normalized:
            return ""

        dist_name = cls._extract_dist_name(normalized)
        canonical = cls._canonical_dist_name(dist_name)
        mapped = cls._MODULE_PACKAGE_ALIASES.get(canonical)
        if mapped and dist_name:
            normalized = re.sub(
                rf"^\s*{re.escape(dist_name)}",
                mapped,
                normalized,
                count=1,
                flags=re.IGNORECASE
            )
            dist_name = cls._extract_dist_name(normalized)
            canonical = cls._canonical_dist_name(dist_name)

        if canonical in cls._OPENCV_PACKAGE_NAMES and canonical != cls._PREFERRED_OPENCV_PACKAGE:
            normalized = cls._replace_requirement_dist_name(normalized, cls._PREFERRED_OPENCV_PACKAGE)
        return normalized

    @classmethod
    def _validate_requirement_spec(cls, requirement: str):
        req = str(requirement or "").strip()
        if not req:
            raise RuntimeError("Empty dependency spec is not allowed")
        if req.startswith("-"):
            raise RuntimeError(f"Unsafe dependency option is not allowed: {req}")
        # 仅接受标准 requirement 形式，不接受 URL/VCS/本地路径安装。
        if "://" in req or req.startswith(("git+", ".", "/")) or "\\" in req:
            raise RuntimeError(f"Unsupported dependency source in sandbox install: {req}")

        if Requirement is not None:
            try:
                parsed = Requirement(req)
                if not parsed.name or parsed.name.startswith("-"):
                    raise RuntimeError(f"Invalid dependency spec: {req}")
            except InvalidRequirement as e:
                raise RuntimeError(f"Invalid dependency spec: {req}") from e

    @classmethod
    def _get_distribution_candidates(cls, dist_name: str) -> list[str]:
        candidates: list[str] = []
        seen: set[str] = set()

        def add(name: str | None):
            if not name:
                return
            canonical_name = cls._canonical_dist_name(name)
            if not canonical_name or canonical_name in seen:
                return
            seen.add(canonical_name)
            candidates.append(name)

        canonical = cls._canonical_dist_name(dist_name)
        if canonical in cls._OPENCV_PACKAGE_NAMES and canonical != cls._PREFERRED_OPENCV_PACKAGE:
            add(cls._PREFERRED_OPENCV_PACKAGE)
        add(dist_name)
        add(cls._MODULE_PACKAGE_ALIASES.get(canonical))

        for item in cls._DIST_COMPAT_GROUPS.get(canonical, ()):
            add(item)

        mapped = cls._MODULE_PACKAGE_ALIASES.get(canonical)
        mapped_canonical = cls._canonical_dist_name(mapped) if mapped else ""
        if mapped_canonical and mapped_canonical != canonical:
            for item in cls._DIST_COMPAT_GROUPS.get(mapped_canonical, ()):
                add(item)

        return candidates

    @classmethod
    def _resolve_import_from_installed_dist(cls, dist_name: str) -> str | None:
        if not dist_name:
            return None

        candidates = cls._get_distribution_candidates(dist_name)

        for candidate in candidates:
            try:
                dist = importlib_metadata.distribution(candidate)
                top_level = dist.read_text("top_level.txt") or ""
                for line in top_level.splitlines():
                    mod = line.strip()
                    if cls._is_identifier(mod):
                        return mod
            except importlib_metadata.PackageNotFoundError:
                continue
            except Exception:
                continue

        try:
            canonical_targets = {cls._canonical_dist_name(c) for c in candidates}
            for mod, dists in importlib_metadata.packages_distributions().items():
                if any(cls._canonical_dist_name(d) in canonical_targets for d in (dists or [])):
                    if cls._is_identifier(mod):
                        return mod
        except Exception:
            pass
        return None

    @classmethod
    def _resolve_import_name(cls, name: str) -> str | None:
        candidate = str(name or "").strip()
        if not candidate:
            return None

        if importlib.util.find_spec(candidate):
            return candidate

        canonical = cls._canonical_dist_name(candidate)
        alias = cls._PKG_IMPORT_ALIASES.get(canonical)
        if alias and importlib.util.find_spec(alias):
            return alias

        dist_resolved = cls._resolve_import_from_installed_dist(candidate)
        if dist_resolved and importlib.util.find_spec(dist_resolved):
            return dist_resolved

        fallback = candidate.replace("-", "_")
        if importlib.util.find_spec(fallback):
            return fallback
        return None

    @classmethod
    def _get_installed_version(cls, dist_name: str, refresh: bool = False) -> str | None:
        if not dist_name:
            return None

        canonical = cls._canonical_dist_name(dist_name)
        if not refresh and canonical in cls._installed_dist_versions:
            return cls._installed_dist_versions[canonical]

        candidates = cls._get_distribution_candidates(dist_name)

        for candidate in candidates:
            try:
                version = importlib_metadata.version(candidate)
                candidate_canonical = cls._canonical_dist_name(candidate)
                cls._installed_packages.add(canonical)
                cls._installed_packages.add(candidate_canonical)
                cls._installed_dist_versions[canonical] = version
                cls._installed_dist_versions[candidate_canonical] = version
                return version
            except importlib_metadata.PackageNotFoundError:
                continue
            except Exception:
                continue
        return None

    @classmethod
    def _requirement_satisfied(cls, requirement: str) -> tuple[bool, str]:
        req_text = str(requirement or "").strip()
        if not req_text:
            return False, ""

        dist_name = ""
        specifier = None
        req_obj = None
        if Requirement is not None:
            try:
                req_obj = Requirement(req_text)
            except InvalidRequirement:
                req_obj = None
            except Exception:
                req_obj = None

        if req_obj is not None:
            if req_obj.marker and not req_obj.marker.evaluate():
                return True, req_obj.name
            if req_obj.extras:
                return False, req_obj.name
            dist_name = req_obj.name
            specifier = req_obj.specifier
        else:
            dist_name = cls._extract_dist_name(req_text)

        if not dist_name:
            return False, ""

        canonical_dist = cls._canonical_dist_name(dist_name)
        if canonical_dist in cls._OPENCV_PACKAGE_NAMES:
            preferred_version = cls._get_installed_version(cls._PREFERRED_OPENCV_PACKAGE)
            if preferred_version is not None and importlib.util.find_spec("cv2"):
                return True, cls._PREFERRED_OPENCV_PACKAGE

        installed_version = cls._get_installed_version(dist_name)
        if installed_version is None:
            return False, dist_name

        if specifier:
            try:
                if installed_version not in specifier:
                    return False, dist_name
            except Exception:
                return False, dist_name
        return True, dist_name

    @classmethod
    def _parse_requirement(cls, requirement: str):
        if Requirement is None:
            return None
        try:
            return Requirement(str(requirement or "").strip())
        except Exception:
            return None

    @staticmethod
    def _requirement_marker_applies(req_obj, selected_extras: set[str] | None = None) -> bool:
        marker = getattr(req_obj, "marker", None)
        if marker is None:
            return True

        extras = {str(extra).strip() for extra in (selected_extras or set()) if str(extra).strip()}
        if not extras:
            try:
                return bool(marker.evaluate({"extra": ""}))
            except Exception:
                return True

        for extra in extras:
            try:
                if marker.evaluate({"extra": extra}):
                    return True
            except Exception:
                continue
        try:
            return bool(marker.evaluate({"extra": ""}))
        except Exception:
            return False

    @classmethod
    def _requirement_extras(cls, requirement: str) -> set[str]:
        req_obj = cls._parse_requirement(requirement)
        return {str(extra).strip() for extra in (getattr(req_obj, "extras", None) or set()) if str(extra).strip()}

    @classmethod
    def _iter_required_dependencies(cls, dist_name: str, selected_extras: set[str] | None = None) -> list[str]:
        try:
            requirement_lines = importlib_metadata.requires(dist_name) or []
        except importlib_metadata.PackageNotFoundError:
            return []
        except Exception:
            return []

        dependencies: list[str] = []
        for req_text in requirement_lines:
            req_obj = cls._parse_requirement(req_text)
            if req_obj is None:
                continue
            if not cls._requirement_marker_applies(req_obj, selected_extras):
                continue
            dependencies.append(str(req_obj))
        return dependencies

    @classmethod
    def _candidate_import_names_for_dist(cls, dist_name: str) -> set[str]:
        names: set[str] = set()
        if not dist_name:
            return names

        canonical = cls._canonical_dist_name(dist_name)
        alias = cls._PKG_IMPORT_ALIASES.get(canonical)
        if alias:
            names.add(alias)

        resolved = cls._resolve_import_from_installed_dist(dist_name)
        if resolved:
            names.add(resolved)

        if cls._is_identifier(dist_name):
            names.add(dist_name)

        normalized = dist_name.replace("-", "_")
        if cls._is_identifier(normalized):
            names.add(normalized)

        module_alias = cls._MODULE_PACKAGE_ALIASES.get(canonical)
        if module_alias:
            mod_norm = module_alias.replace("-", "_")
            if cls._is_identifier(mod_norm):
                names.add(mod_norm)
        return names

    @classmethod
    def release_device_memory(cls):
        try:
            torch_mod = importlib.import_module("torch")
        except Exception:
            return
        try:
            if torch_mod.cuda.is_available():
                torch_mod.cuda.empty_cache()
        except Exception:
            pass
        try:
            if hasattr(torch_mod, "mps") and hasattr(torch_mod.mps, "empty_cache"):
                torch_mod.mps.empty_cache()
        except Exception:
            pass
        try:
            if hasattr(torch_mod, "xpu") and hasattr(torch_mod.xpu, "empty_cache"):
                torch_mod.xpu.empty_cache()
        except Exception:
            pass

    def _best_effort_release_for_install(self, dist_name: str):
        if not dist_name:
            return

        module_roots = self._candidate_import_names_for_dist(dist_name)
        if not module_roots:
            return

        for mod_name in module_roots:
            self._base_namespace.pop(mod_name, None)

        purge_prefixes = tuple(sorted(module_roots, key=len, reverse=True))
        for key in list(sys.modules.keys()):
            if key in module_roots or key.startswith(purge_prefixes):
                sys.modules.pop(key, None)

        self.release_device_memory()
        gc.collect()

    def load_core_optional_modules(self):
        for alias, module_name in self._CORE_OPTIONAL_MODULES.items():
            if alias in self._base_namespace:
                continue
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            self._base_namespace[alias] = module

    def get_allowed_import_prefixes(self) -> set[str]:
        prefixes: set[str] = set()

        def add(name: str | None):
            token = str(name or "").strip()
            if not token:
                return
            prefixes.add(token)
            if "." in token:
                prefixes.add(token.split(".", maxsplit=1)[0])

        for module_name in self._PKG_IMPORT_ALIASES.values():
            add(module_name)
        for module_name in self._CORE_OPTIONAL_MODULES.values():
            add(module_name)

        for value in self._base_namespace.values():
            if isinstance(value, ModuleType):
                add(getattr(value, "__name__", ""))

        return prefixes

    @classmethod
    def _build_pip_install_command(
        cls,
        package: str,
        constraint_path: str | None = None,
        no_deps: bool = False
    ) -> list[str]:
        command = [sys.executable, "-m", "pip", "install"]
        if no_deps:
            command.append("--no-deps")
        elif constraint_path:
            command.extend(["--constraint", constraint_path])
        command.append(package)
        return command

    @classmethod
    def _opencv_constraint_text(cls) -> str:
        return "\n".join(f"{package}<0" for package in cls._BLOCKED_OPENCV_PACKAGES) + "\n"

    @classmethod
    def _pip_error_mentions_blocked_opencv(cls, output: str) -> bool:
        text = str(output or "").lower()
        return any(package in text for package in cls._BLOCKED_OPENCV_PACKAGES)

    @staticmethod
    def _pip_output_tail(output: str, limit: int = 4000) -> str:
        text = str(output or "").strip()
        if len(text) <= limit:
            return text
        return text[-limit:]

    def _run_pip_install(self, package: str, no_deps: bool = False) -> subprocess.CompletedProcess:
        constraint_path = None
        if not no_deps:
            constraint_file = tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                suffix=".txt",
                prefix="chatimageenhance-opencv-constraints-",
                delete=False,
            )
            try:
                with constraint_file:
                    constraint_file.write(self._opencv_constraint_text())
                constraint_path = constraint_file.name
            except Exception:
                try:
                    constraint_file.close()
                except Exception:
                    pass
                raise

        command = self._build_pip_install_command(package, constraint_path, no_deps=no_deps)
        try:
            return subprocess.run(command, capture_output=True, text=True)
        finally:
            if constraint_path:
                try:
                    os.unlink(constraint_path)
                except OSError:
                    pass

    def _install_package_selective_deps(
        self,
        package: str,
        visiting: set[str] | None = None
    ):
        """
        Install package with --no-deps, then recursively install metadata deps.

        This is used when pip's resolver rejects a package because one of its
        transitive dependencies pins an OpenCV wheel variant. We keep the
        existing opencv-contrib-python and still install the rest of the chain.
        """
        package = self._normalize_package_name(str(package))
        if not package:
            return
        self._validate_requirement_spec(package)

        dist_name = self._extract_dist_name(package)
        if not dist_name:
            return

        canonical = self._canonical_dist_name(dist_name)
        visiting = visiting if visiting is not None else set()
        if canonical in visiting:
            return

        is_satisfied, satisfied_dist = self._requirement_satisfied(package)
        if is_satisfied:
            if satisfied_dist:
                self._installed_packages.add(self._canonical_dist_name(satisfied_dist))
            return

        visiting.add(canonical)
        self._best_effort_release_for_install(dist_name)
        result = self._run_pip_install(package, no_deps=True)
        pip_output = f"{result.stdout or ''}\n{result.stderr or ''}"
        if result.returncode != 0:
            output_tail = self._pip_output_tail(pip_output)
            detail = f" output={output_tail}" if output_tail else ""
            raise RuntimeError(
                "pip dynamic install failed during selective dependency install. "
                f"package={package}; exit_code={result.returncode}.{detail} "
                "If this is a Windows file-lock issue, restart the process and retry."
            )

        canonical = self._canonical_dist_name(dist_name)
        self._installed_packages.add(canonical)
        installed_version = self._get_installed_version(dist_name, refresh=True)
        if installed_version is not None:
            self._installed_dist_versions[canonical] = installed_version

        for dependency in self._iter_required_dependencies(
            dist_name,
            selected_extras=self._requirement_extras(package)
        ):
            normalized_dependency = self._normalize_package_name(dependency)
            if not normalized_dependency:
                continue
            self._validate_requirement_spec(normalized_dependency)
            dep_satisfied, dep_dist = self._requirement_satisfied(normalized_dependency)
            if dep_satisfied:
                if dep_dist:
                    self._installed_packages.add(self._canonical_dist_name(dep_dist))
                continue
            self._install_package_selective_deps(normalized_dependency, visiting)

        visiting.discard(canonical)

    def install_packages(self, packages: Iterable[str] | None):
        if packages is None:
            return

        for package in packages:
            package = self._normalize_package_name(str(package))
            if not package:
                continue
            self._validate_requirement_spec(package)

            dist_name = self._extract_dist_name(package)

            is_satisfied, satisfied_dist = self._requirement_satisfied(package)
            if is_satisfied:
                if satisfied_dist:
                    self._installed_packages.add(self._canonical_dist_name(satisfied_dist))
                continue

            self._best_effort_release_for_install(dist_name)
            result = self._run_pip_install(package)
            pip_output = f"{result.stdout or ''}\n{result.stderr or ''}"
            if result.returncode != 0 and self._pip_error_mentions_blocked_opencv(pip_output):
                self._install_package_selective_deps(package)
                continue

            if result.returncode != 0:
                output_tail = self._pip_output_tail(pip_output)
                detail = f" output={output_tail}" if output_tail else ""
                raise RuntimeError(
                    "pip dynamic install failed. "
                    f"package={package}; exit_code={result.returncode}.{detail} "
                    "If this is a Windows file-lock issue, restart the process and retry."
                )

            if dist_name:
                canonical = self._canonical_dist_name(dist_name)
                self._installed_packages.add(canonical)
                installed_version = self._get_installed_version(dist_name, refresh=True)
                if installed_version is not None:
                    self._installed_dist_versions[canonical] = installed_version

    def import_modules(self, imports: Iterable[str] | None):
        if imports is None:
            return

        for imp in imports:
            raw_name = str(imp).strip()
            if not raw_name:
                continue

            mod_name = self._resolve_import_name(raw_name)
            if not mod_name:
                raise RuntimeError(f"Cannot import module {raw_name}")

            module = importlib.import_module(mod_name)
            root_name = mod_name.split(".", maxsplit=1)[0]
            if root_name != mod_name:
                root_module = importlib.import_module(root_name)
                self._base_namespace[root_name] = root_module
            if self._is_identifier(mod_name):
                self._base_namespace[mod_name] = module
            if raw_name != mod_name and self._is_identifier(raw_name):
                self._base_namespace[raw_name] = module

    @staticmethod
    def _extract_missing_module_name(exc: Exception) -> str:
        if isinstance(exc, ModuleNotFoundError):
            name = str(getattr(exc, "name", "") or "").strip()
            if name:
                return name
        msg = str(exc)
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", msg)
        return match.group(1).strip() if match else ""

    @classmethod
    def _guess_package_from_module_name(cls, module_name: str) -> str:
        root = str(module_name or "").strip().split(".", maxsplit=1)[0]
        if not root:
            return ""
        canonical_root = cls._canonical_dist_name(root)
        if canonical_root in cls._AUTO_INSTALL_MODULE_PACKAGE_ALIASES:
            return cls._AUTO_INSTALL_MODULE_PACKAGE_ALIASES[canonical_root]
        if canonical_root in cls._MODULE_PACKAGE_ALIASES:
            return cls._MODULE_PACKAGE_ALIASES[canonical_root]
        if re.fullmatch(r"[A-Za-z0-9_]+", root):
            return root.replace("_", "-")
        return ""

    def recover_missing_module_dependency(self, exc: Exception) -> bool:
        missing_module = self._extract_missing_module_name(exc)
        if not missing_module:
            return False

        root_module = missing_module.split(".", maxsplit=1)[0].strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", root_module):
            return False
        if root_module in self._auto_install_failed:
            return False
        if root_module in self._auto_install_attempted:
            return False

        package = self._guess_package_from_module_name(root_module)
        if not package:
            return False

        self._auto_install_attempted.add(root_module)
        try:
            self.install_packages([package])
            self.load_core_optional_modules()
            return True
        except Exception:
            self._auto_install_failed.add(root_module)
            return False

    def extend_runtime(
        self,
        additional_imports: Iterable[str] | None = None,
        additional_packages: Iterable[str] | None = None
    ):
        self.install_packages(additional_packages)
        self.load_core_optional_modules()
        self.import_modules(additional_imports)
