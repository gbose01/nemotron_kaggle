"""Blackwell / Kaggle environment hardening for Nemotron-3-Nano-30B.

Consolidates every workaround needed to load and train the full BF16 model on an
RTX PRO 6000 (Blackwell) Kaggle notebook with Internet OFF. Import and call
``apply()`` BEFORE loading the model, then ``disable_fast_path(model)`` and
``freeze_routers(model)`` AFTER the model is built.

Patches applied by ``apply()``:
  1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (reduce fragmentation).
  2. Pure-PyTorch rmsnorm_fn replacement injected into all loaded modules.
  3. Mock cutlass.*/quack.* modules + meta-path finder (Nemotron custom code
     imports them but the CUDA kernels are unavailable / broken on Blackwell).
  4. Triton ptxas-blackwell fix (full bin dir copy + env vars + version stub).

This single module is the source of truth shared by ``train_blackwell.py`` and
``scripts/setup_blackwell_offline.py`` so the cell-based notebook and the offline
script never drift apart.
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
import types
from importlib.machinery import ModuleSpec


# Candidate locations for the Blackwell ptxas binary. The Kaggle utility script
# ("nvidia-utility-script") ships a prebuilt one; the offline setup copies the
# competition image's copy to /opt/nvidia.
_PTXAS_CANDIDATES = [
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/triton/backends/nvidia/bin/ptxas-blackwell",
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/triton/backends/nvidia/bin/ptxas-blackwell",
    "/opt/nvidia/ptxas-blackwell",
]

_MOCK_MODULES = [
    "cutlass", "cutlass.cute", "cutlass.cutlass_dsl",
    "cutlass._mlir", "cutlass._mlir.dialects",
    "quack", "quack.rmsnorm", "quack.softmax", "quack.cross_entropy",
    "quack.utils", "quack.copy_utils", "quack.layout_utils",
    "quack.compile_utils", "quack.cute_dsl_utils",
]


def set_alloc_conf() -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _pure_rmsnorm_fn(x, weight, bias=None, z=None, eps=1e-5,
                     group_size=None, norm_before_gate=True, upcast=True):
    """Pure-PyTorch RMSNorm matching the mamba_ssm/Nemotron signature.

    Avoids the Triton/CUTLASS fused kernel that fails to compile on Blackwell.
    """
    import torch
    import torch.nn.functional as F

    dtype = x.dtype
    if upcast:
        x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    out = x_normed * weight.float()
    if bias is not None:
        out = out + bias.float()
    if z is not None:
        out = out * F.silu(z.float())
    return out.to(dtype)


def patch_rmsnorm() -> int:
    """Replace ``rmsnorm_fn`` on every already-imported module that exposes it."""
    patched = 0
    for name, mod in list(sys.modules.items()):
        if mod is not None and hasattr(mod, "rmsnorm_fn"):
            try:
                mod.rmsnorm_fn = _pure_rmsnorm_fn
                patched += 1
            except Exception:
                pass
    return patched


class _CutlassMock:
    """Permissive stand-in that swallows attribute access, calls and operators."""

    def __getattr__(self, name):
        return _CutlassMock()

    def __call__(self, *args, **kwargs):
        return _CutlassMock()

    def __iter__(self):
        return iter([])

    def __bool__(self):
        return False

    def __or__(self, other):
        return _CutlassMock()

    def __ror__(self, other):
        return _CutlassMock()

    def __class_getitem__(cls, item):
        return cls

    def __getitem__(self, item):
        return _CutlassMock()


class _MockModule(types.ModuleType):
    """Module that returns mocks for normal attrs but raises on dunders so that
    ``inspect`` / ``importlib`` machinery keeps working."""

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return _CutlassMock()


def _make_mock(name: str) -> types.ModuleType:
    mod = _MockModule(name)
    mod.__path__ = []
    mod.__package__ = name
    mod.__spec__ = ModuleSpec(name, None, is_package=True)
    sys.modules[name] = mod
    return mod


class _MockImportFinder:
    """Catches any cutlass.* / quack.* import we did not pre-register."""

    MOCK_PREFIXES = ("cutlass", "quack")

    def find_spec(self, fullname, path, target=None):
        for prefix in self.MOCK_PREFIXES:
            if fullname == prefix or fullname.startswith(prefix + "."):
                if fullname not in sys.modules:
                    _make_mock(fullname)
                return sys.modules[fullname].__spec__
        return None


def install_cutlass_mocks() -> None:
    for mod_name in _MOCK_MODULES:
        if mod_name not in sys.modules:
            _make_mock(mod_name)
    if not any(isinstance(f, _MockImportFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _MockImportFinder())


def _make_executable(path: str) -> None:
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def apply_triton_fix() -> bool:
    """Copy the Blackwell ptxas binary into place and point Triton at it.

    Returns True if a fix was applied, False if no ptxas-blackwell was found.
    """
    src = next((p for p in _PTXAS_CANDIDATES if os.path.exists(p)), None)
    if src is None:
        return False

    dst = "/tmp/ptxas-blackwell"
    shutil.copy2(src, dst)
    _make_executable(dst)

    os.environ["TRITON_PTXAS_PATH"] = dst
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = dst

    # Copy the full triton nvidia bin dir so companion tools resolve too.
    try:
        import triton.backends.nvidia as nv_backend

        src_bin = os.path.join(os.path.dirname(nv_backend.__file__), "bin")
        dst_bin = "/tmp/triton_nvidia_bin"
        if os.path.isdir(src_bin):
            shutil.copytree(src_bin, dst_bin, dirs_exist_ok=True)
            for f in os.listdir(dst_bin):
                fp = os.path.join(dst_bin, f)
                if os.path.isfile(fp):
                    _make_executable(fp)
    except Exception:
        pass

    # ptxas-blackwell reports an unusual version; stub it so Triton accepts it.
    try:
        import triton.backends.nvidia.compiler as nv_compiler

        nv_compiler.get_ptxas_version = lambda *a, **k: (12, 9, 0)
    except Exception:
        pass

    return True


def disable_fast_path(model) -> int:
    """Force the stable (non-Triton) path on every submodule that supports it."""
    count = 0
    for _, mod in model.named_modules():
        if hasattr(mod, "is_fast_path_available"):
            mod.is_fast_path_available = False
            count += 1
    return count


def freeze_routers(model) -> int:
    """Freeze MoE router params for training stability (no-op if none exist)."""
    frozen = 0
    for name, param in model.named_parameters():
        if "router" in name.lower():
            param.requires_grad = False
            frozen += 1
    return frozen


def apply(verbose: bool = True) -> None:
    """Apply all pre-model-load patches. Call once, before from_pretrained."""
    set_alloc_conf()
    install_cutlass_mocks()
    n_rms = patch_rmsnorm()
    triton_ok = apply_triton_fix()
    if verbose:
        print("[blackwell_env] PYTORCH_CUDA_ALLOC_CONF =",
              os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
        print(f"[blackwell_env] cutlass/quack mocks installed ({len(_MOCK_MODULES)} preregistered)")
        print(f"[blackwell_env] patched rmsnorm_fn on {n_rms} module(s)")
        print("[blackwell_env] Triton ptxas fix:",
              "applied" if triton_ok else "skipped (no ptxas-blackwell found)")


if __name__ == "__main__":
    apply()
