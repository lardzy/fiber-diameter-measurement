"""Collect the compiled diameter backend for all FDM executables and probes."""
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, copy_metadata, get_module_file_attribute


hiddenimports = ["skimage.morphology"]
datas = collect_data_files("skimage", includes=["**/*.pyi"])
for distribution_name in ("scikit-image", "scipy", "lazy-loader"):
    datas += copy_metadata(distribution_name)

# SciPy moved its vendored array API from _lib to _external in 1.18. The numpy
# adapter imports these names dynamically; older PyInstaller hooks cover only
# the old location. Detect the installed layout (Python 3.11 still uses 1.17).
scipy_root = Path(get_module_file_attribute("scipy")).parent
for parent in ("_lib", "_external"):
    if (scipy_root / parent / "array_api_compat" / "numpy").is_dir():
        hiddenimports += [
            f"scipy.{parent}.array_api_compat.numpy.fft",
            f"scipy.{parent}.array_api_compat.numpy.linalg",
        ]
