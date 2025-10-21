# VS Code Integration

- Open the Extensions view and install **C/C++** (Microsoft) if it is not already present; it reads the new `.clang-format` automatically.
- Go to *File → Preferences → Settings*, search for **Format On Save**, and enable it so VS Code uses `.clang-format` each time you save.
- In the same Settings window, search for **Clang Format Style** and set it to `file` so the editor relies on the repository configuration.
- Under *C/C++ › Code Analysis*, enable **Clang Tidy** and leave the executable field empty; the extension will invoke the compiler-provided tool and respect `.clang-tidy`.
- Still under Code Analysis, ensure **Clang Tidy: Use `--config=`** is enabled so the checks in `.clang-tidy` drive diagnostics.
- `.editorconfig` is picked up automatically; double-check *Text Editor › Files* that **Insert Final Newline** and **Trim Trailing Whitespace** are enabled to mirror the shared settings.
