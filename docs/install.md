# Installing MATS from scratch

This guide starts from a computer with nothing set up: no Python, no Git, no
package manager. If you already have Python 3.10 or newer and Git LFS, the short
version in the [README](../README.md#installation) is enough.

Work through the steps in order. Each one says what you should see when it has
worked. Type commands one at a time. The `$` or `%` at the start of a line in
other guides is the prompt, not part of the command.

---

## 1. What you need

- **Python 3.10, 3.11, 3.12, or 3.13.** Python 3.13 is recommended. Python 3.9
  and older will not work, and that includes the `python3` built into macOS,
  which is 3.9.
- **Git** and **Git LFS.** Git LFS delivers the marker-detection model, which
  every run needs.
- **About 2 GB of free disk space** for MATS and its packages, and internet
  access to `github.com` and `pypi.org`.
- **Administrator rights** make things simpler. Without them, follow
  [No administrator rights](#no-administrator-rights-any-operating-system).

Supported computers:

| Computer | Status |
|---|---|
| Mac with Apple silicon (M1 or newer) | Supported, Python 3.10–3.13 |
| Windows, 64-bit Intel/AMD | Supported, Python 3.10–3.13 |
| Linux, x86_64 or ARM64 | Supported, Python 3.10–3.13 |
| Mac with an Intel processor | Python 3.10–3.12 only, with an extra install flag (see [Intel Macs](#intel-macs)). Untested |
| Windows on ARM (Snapdragon, Surface Pro X) | Not supported. PyTorch publishes no Windows-on-ARM packages on PyPI |

On a Mac, **Apple menu → About This Mac** shows *Chip: Apple M…* (Apple
silicon) or *Processor: Intel*.

---

## 2. Check what you already have

Open a terminal (macOS: **Terminal**; Windows: **PowerShell**; Linux: your
terminal app) and run:

```bash
python3 --version        # Windows: py --version
git --version
git lfs version
```

| Result | Meaning |
|---|---|
| `Python 3.10.x` to `Python 3.13.x` | Good. Skip to [step 4](#4-install-git-lfs) |
| `Python 3.9.x` or older | Too old. Install Python 3.13 ([step 3](#3-install-python-313)) |
| `command not found`, or Windows opens the Microsoft Store | No Python yet. Install Python 3.13 ([step 3](#3-install-python-313)) |
| `git version …` | Git is installed |
| `git-lfs/3.x …` | Git LFS is installed |
| `git: 'lfs' is not a git command` | Git LFS is missing ([step 4](#4-install-git-lfs)) |

**On a new Mac,** the first `python3` or `git` command opens a dialog offering
to install Apple's *command line developer tools*. Click **Install**, wait for it
to finish, then run the command again. This gives you Git. The `python3` it
installs is 3.9, which is too old for MATS, so you still need step 3.

---

## 3. Install Python 3.13

python.org publishes installers only while a Python version still gets bug
fixes. If the newest release listed for a version says *"No files for this
release"*, scroll down to the newest one that has an installer.

### macOS

**Recommended: the python.org installer.** No Homebrew needed.

1. Go to <https://www.python.org/downloads/macos/> and download the latest
   **Python 3.13** "macOS 64-bit universal2 installer".
2. Open the downloaded `.pkg` and click through. It asks for your Mac password.
3. When it finishes, open **Applications → Python 3.13** and double-click
   **Install Certificates.command**. This lets Python make secure downloads.
4. Close the terminal window, open a new one, and check:

   ```bash
   python3.13 --version     # should print Python 3.13.x
   ```

**Alternative: Homebrew.** If you already use Homebrew, or want to:

1. Install Homebrew from <https://brew.sh> (it asks for your Mac password; the
   password doesn't show as you type).
2. **Run the lines Homebrew prints under "Next steps".** Until you do, the
   terminal replies `command not found: brew`. On Apple silicon they look like:

   ```bash
   echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
   eval "$(/opt/homebrew/bin/brew shellenv)"
   ```

3. Install Python and Git LFS together:

   ```bash
   brew install python@3.13 git-lfs
   python3.13 --version     # should print Python 3.13.x
   ```

#### Intel Macs

PyTorch's last release for Intel Macs is 2.2, which supports Python up to 3.12
and needs NumPy 1.x. Install **Python 3.12** instead of 3.13 (python.org's last
3.12 installer is **3.12.10**; Homebrew: `brew install python@3.12`), use
`python3.12` wherever this guide says `python3.13`, and add `"numpy<2"` to the
install command in [step 5](#5-download-and-install-mats):

```bash
python -m pip install -e ".[app]" "numpy<2"
```

We have not tested MATS on an Intel Mac.

### Windows

1. Go to <https://www.python.org/downloads/windows/> and download the latest
   **Python 3.13** "Windows installer (64-bit)".
2. On the installer's first screen, tick **Add python.exe to PATH**, then click
   **Install Now**.
3. Open a new PowerShell window and check:

   ```powershell
   py -3.13 --version       # should print Python 3.13.x
   ```

If typing `python` opens the Microsoft Store, that's a Windows placeholder, not
Python. Use `py -3.13` as shown.

### Linux

Many distributions already ship a Python that works (any of 3.10–3.13). Check
with `python3 --version`.

- **Ubuntu 22.04 or newer, Debian 12 or newer:**

  ```bash
  sudo apt install python3-venv python3-pip git git-lfs
  ```

- **Fedora:** `sudo dnf install python3 git git-lfs`
- **RHEL, Rocky, or Alma 8 or 9:** `sudo dnf install python3.12 git git-lfs`, then
  use `python3.12` wherever this guide says `python3.13`.

If your distribution's Python is older than 3.10 or you can't use `sudo`, use
[Miniforge](#no-administrator-rights-any-operating-system).

### No administrator rights (any operating system)

Miniforge installs Python, Git, and Git LFS inside your own user folder, with no
administrator password.

1. Download and run the installer for your system from
   <https://github.com/conda-forge/miniforge>. On macOS or Linux:

   ```bash
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```

   Answer **yes** when it offers to initialise your shell, then open a new
   terminal. On Windows, run `Miniforge3-Windows-x86_64.exe`, choose
   **Just Me**, and afterwards use the **Miniforge Prompt** from the Start menu.

2. Create an environment containing Python 3.13, Git, and Git LFS:

   ```bash
   conda create -n mats -c conda-forge python=3.13 git git-lfs
   conda activate mats
   ```

3. Continue with step 4. In step 5, **skip the two virtual-environment lines**:
   the conda environment already plays that role. Run `conda activate mats` in
   each new terminal instead of activating `.venv`.

---

## 4. Install Git LFS

Skip this if `git lfs version` already worked, or if you installed Git LFS with
Homebrew, `apt`/`dnf`, or Miniforge above.

- **macOS without Homebrew:** download the macOS package from
  <https://git-lfs.com>, unzip it, and in the unzipped folder run
  `sudo ./install.sh`.
- **Windows:** install [Git for Windows](https://gitforwindows.org), which
  includes Git LFS. The default options are fine.

Then, once per computer:

```bash
git lfs install          # prints "Git LFS initialized."
```

`git lfs install` is all you need. If a guide suggests
`git lfs install --system`, skip it: that variant needs administrator rights,
and MATS doesn't need it.

---

## 5. Download and install MATS

Choose a folder that iCloud Drive, OneDrive, or Dropbox doesn't sync, such as
your home folder. The installed packages are thousands of files, and a sync
service can upload them or remove local copies.

**macOS and Linux:**

```bash
cd ~
git clone https://github.com/Breeding-Insight/Morphometric-Analysis-Toolbox-for-Segmentation.git
cd Morphometric-Analysis-Toolbox-for-Segmentation
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[app]"
mats doctor
```

If `python3 --version` already reported 3.10–3.13, you can use `python3` in
place of `python3.13`.

**Windows (PowerShell):**

```powershell
cd ~
git clone https://github.com/Breeding-Insight/Morphometric-Analysis-Toolbox-for-Segmentation.git
cd Morphometric-Analysis-Toolbox-for-Segmentation
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[app]"
mats doctor
```

What each part does:

- **`git clone`** downloads MATS together with the ~134 MB marker model. Check
  it with `ls -l weights/rf_detr_marker.pth` (Windows: `dir weights`): it should
  be about 134,000,000 bytes. `ls -lh` shows this as `128M`, which is correct.
  A file of about 134 **bytes** means Git LFS wasn't set up. Run
  `git lfs install && git lfs pull --exclude="weights/birefnet_leaf.pth"`.
- **`python -m venv .venv`** creates a *virtual environment*, a private copy of
  Python inside the MATS folder. Everything MATS installs goes there, so it
  can't clash with other software, and it needs no administrator rights.
  `source .venv/bin/activate` (Windows: `.venv\Scripts\Activate.ps1`) switches
  the terminal to it. Your prompt then starts with `(.venv)`.
- **`python -m pip`** runs the installer that belongs to the active Python.
  Plain `pip` is often missing, or belongs to a different Python.
- **`-e`** (editable install) matters: it lets MATS find the model in the
  folder you cloned. Don't leave it out.
- **The install downloads packages from PyPI,** mostly PyTorch. Expect several
  minutes.

**It worked if** `mats doctor` prints your Python version and
`environment: virtual environment` (or `conda environment`), shows
`RF-DETR checkpoint   [ok]`, and ends without "A required weight is missing".

Next: [Running MATS](../README.md#running-mats). `mats app` opens the app in your
browser.

---

## 6. Every time you open a new terminal

The virtual environment is active only in the terminal where you activated it.
Before using `mats` in a new terminal:

```bash
cd ~/Morphometric-Analysis-Toolbox-for-Segmentation
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
                                 # Miniforge: conda activate mats
```

---

## 7. If something goes wrong

Find the message you see in the left column.

### While installing

| You see | Why | Fix |
|---|---|---|
| `xcode-select: note: No developer tools were found, requesting install.` | macOS is installing its command line tools, which include Git | Click **Install** in the dialog, wait, and run the command again |
| `Python 3.9.6`, or any version below 3.10 | This is the Python built into macOS (or your system), and it's too old | Install Python 3.13 ([step 3](#3-install-python-313)) and use `python3.13` |
| *"No files for this release"* on python.org | That Python version no longer gets installers | Pick the newest release of that version that has one ([step 3](#3-install-python-313)) |
| `command not found: brew`, right after installing Homebrew | Homebrew isn't on your PATH yet | Run the "Next steps" lines Homebrew printed ([step 3](#macos)) |
| `Error: python not installed` from `brew upgrade python` | Homebrew has no Python of its own to upgrade | `brew install python@3.13` |
| `command not found: pip` or `pip3` | No `pip` outside a virtual environment | Activate the virtual environment and use `python -m pip` |
| `File "setup.py" or "setup.cfg" not found. Directory cannot be installed in editable mode` | The `pip` that came with the old system Python is too old | Use a Python 3.13 virtual environment ([step 5](#5-download-and-install-mats)), which has a current `pip` |
| `Defaulting to user installation because normal site-packages is not writeable` | You're installing into the system Python, not a virtual environment | Press Ctrl-C, then create and activate the virtual environment ([step 5](#5-download-and-install-mats)) |
| `requires a different Python: 3.9.6 not in '>=3.10'` | The Python in use is too old | Install Python 3.13 and recreate the virtual environment with it |
| `No matching distribution found for rfdetr==1.5.2` | Python older than 3.10; older copies of MATS fail here instead of with the message above | Same as above |
| `No matching distribution found for torch` | No PyTorch for this computer and Python: an Intel Mac with Python 3.13+, or Windows on ARM | Intel Mac: use Python 3.12 ([Intel Macs](#intel-macs)). Windows on ARM isn't supported |
| `error: externally-managed-environment` | Homebrew's and some Linux systems' Python refuse installs outside a virtual environment | Create and activate the virtual environment ([step 5](#5-download-and-install-mats)) |
| `ensurepip is not available` when creating `.venv` | Debian/Ubuntu split virtual-environment support into its own package | `sudo apt install python3-venv` (or the `python3.X-venv` package the message names) |
| `mats: command not found` | The virtual environment isn't active in this terminal | Activate it ([step 6](#6-every-time-you-open-a-new-terminal)) |
| `could not lock config file /etc/gitconfig: Permission denied` after `git lfs install --system` | `--system` needs administrator rights | Not needed. `git lfs install` alone is enough |
| `ls -lh weights/rf_detr_marker.pth` shows `128M` | That's the correct size (134 MB = 128 MiB) | Nothing to fix |
| `weights/rf_detr_marker.pth` is ~134 bytes | You cloned before Git LFS was set up | `git lfs install && git lfs pull --exclude="weights/birefnet_leaf.pth"` |
| `Activate.ps1 cannot be loaded because running scripts is disabled on this system` | PowerShell's default script policy | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again. Or use Command Prompt and `.venv\Scripts\activate.bat` |
| `CERTIFICATE_VERIFY_FAILED` during the install | Python can't verify secure connections | python.org on macOS: run **Install Certificates.command** ([step 3](#macos)). On an institutional network, ask IT for its proxy settings |

### Messages that are safe to ignore

These appear during a normal run and don't affect the measurements.

| You see | What it is |
|---|---|
| `[WARNING] rf-detr - Using patch size 16 instead of 14 …` / `… positional encodings than DINOv2 …` / `Model is not optimized for inference …` | Routine notes from the marker-detection library as it loads the model |
| `UserWarning: Error fetching version info <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] …>` from `albumentations` | A library's check for its own updates failed. On python.org Python for macOS, **Install Certificates.command** ([step 3](#macos)) makes it go away |
| `FutureWarning: torch.jit.script is deprecated` | A PyTorch notice for library developers |
| `For better performance, install the Watchdog module` (from `mats app`) | Optional. It only speeds up reloading while editing the app's code |

If you're stuck, run `mats doctor` (once MATS is installed) and include its
output when you ask for help or open an issue at
<https://github.com/Breeding-Insight/Morphometric-Analysis-Toolbox-for-Segmentation/issues>.
An AI coding assistant opened in the MATS folder reads
[AGENTS.md](../AGENTS.md) and can work through this guide with you.
