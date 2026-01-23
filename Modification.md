# Library Modifications

This document outlines necessary patches for `dscribe` and `deeptime` libraries to ensure compatibility with this project.

## 1. Locate Installation Paths

You can use the following Python script to find the exact location of the library files on your system.

```python
import os
import importlib.util

def find_module_path(module_name):
    try:
        spec = importlib.util.find_spec(module_name)
        if spec and spec.origin:
             # spec.origin usually points to __init__.py or the shared object file
            return os.path.dirname(spec.origin)
        return None
    except ImportError:
        return None

def print_locations():
    # 1. dscribe
    dscribe_path = find_module_path("dscribe")
    if dscribe_path:
        target_path = os.path.join(dscribe_path, "descriptors")
        print(f"[dscribe] Descriptors path:\n{target_path}\n")
        print(f"  -> Look for 'soap.py' in this directory.\n")
    else:
        print("[dscribe] Package not found.\n")

    # 2. deeptime
    deeptime_path = find_module_path("deeptime")
    if deeptime_path:
        target_path = os.path.join(deeptime_path, "decomposition", "deep")
        print(f"[deeptime] Deep decomposition path:\n{target_path}\n")
        print(f"  -> Look for '_tae.py' in this directory.\n")
    else:
        print("[deeptime] Package not found.\n")

if __name__ == "__main__":
    print("--- Finding Library Paths ---\n")
    print_locations()
```

## 2. dscribe Patch

**Target:** `dscribe/descriptors/soap.py`

**Description:** Reduce memory usage by casting positions to `float16`.

**Instruction:**
1. Navigate to the `dscribe/descriptors` directory.
2. Open `soap.py`.
3. Locate **line 338** (or nearby).
4. Replace the existing line with:
   ```python
   list_positions.append(system.get_positions()[i].astype(np.float16))
   ```

## 3. deeptime Patch

**Target:** `deeptime/decomposition/deep/_tae.py`

**Description:** Use a custom version of the Time-lagged Autoencoder (TAE).

**Instruction:**
1. Navigate to the `deeptime/decomposition/deep` directory.
2. Replace the existing `_tae.py` file with the version provided in this repository:
   - Source: `./assets/_tae.py`
