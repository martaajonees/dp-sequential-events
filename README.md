<h1 align="center">Differential Privacy in Sequential Event Logging</h1>

<p align="center">
  <img src="https://badgen.net/badge/license/MIT/orange?icon=github" alt="license">
  <img src="https://badgen.net/badge/language/Python/yellow" alt="language">
  <img src="https://badgen.net/badge/build/passing/green?icon=githubactions" alt="build">
  <img src="https://badgen.net/pypi/v/dp-sequential-events" alt="PyPI version">
  <img src="https://img.shields.io/pypi/pyversions/dp-sequential-events?color=red" alt="Python versions">
</p>

<p align="center">
  <picture>
    <source srcset="https://github.com/user-attachments/assets/cfec7311-e4b5-444d-b4ad-fa01b438f985" width="380" media="(prefers-color-scheme: dark)">
    <img src="https://github.com/user-attachments/assets/25125997-a45e-48de-b7bb-e36ea52f0ee1" alt="Logo" width="380">
  </picture>
</p>

<p align="center">
  <a href="https://colab.research.google.com/drive/17jejpDl4sX9L8885Pll4D_PJpxudtFL9#scrollTo=2Xai_2ImKTd9">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Google Colab">
  </a>
</p>

---

## Overview

Sequential event logs often contain sensitive information. `dp-sequential-events` implements **differential privacy (DP)** techniques to anonymize these logs while preserving the statistical properties needed for meaningful analysis.

**Two pipelines are available:**

| Pipeline | Description | Best for |
|---|---|---|
| **Full pipeline** | Anonymizes the complete event log end to end | Publishing or sharing full datasets |
| **Pattern-oriented** | Focuses on frequent behavioral patterns | Analysis where process patterns must be preserved |

---

## Installation

Install via pip:

```bash
pip install dp-sequential-events
```

Launch the interactive interface:

```bash
privseq
```

---

## Usage

Once launched, `privseq` opens a step-by-step interactive menu: no flags to memorize.

<p align="center">
  <picture>
    <source srcset="https://github.com/user-attachments/assets/0a203990-6d15-4fe3-aa20-27bd0f6cec04" width="600" media="(prefers-color-scheme: dark)">
    <img src="https://github.com/user-attachments/assets/2d0e0ff7-a97c-48b6-8bcd-a3d012a5ee45" alt="Usage screenshot" width="600">
  </picture>
</p>

---

## Input format

Your event log should be a CSV file with the following columns:

| Column | Description |
|---|---|
| `CaseID` | Unique case identifier |
| `Activity` | Activity name |
| `Timestamp` | Event timestamp (ISO 8601) |

---

## Authors

<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/martaajonees">
          <img src="https://avatars.githubusercontent.com/u/100365874?v=4" width="80px" style="border-radius:50%" alt="Marta Jones"/><br/>
          <sub><b>Marta Jones</b></sub>
        </a><br/>
        <a href="https://github.com/martaajonees/dp-sequential-events/commits?author=martaajonees" title="Code">💻 Developer</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/ichi91">
          <img src="https://avatars.githubusercontent.com/u/41892183?v=4" width="80px" style="border-radius:50%" alt="Anailys Hernandez"/><br/>
          <sub><b>Anailys Hernandez</b></sub>
        </a><br/>
        <a href="https://github.com/ichi91/Local_Privacy/commits?author=ichi91" title="Method Designer">💡 Method Designer</a>
      </td>
    </tr>
  </tbody>
</table>

---

<p align="center">
  <sub>MIT License · <a href="https://pypi.org/project/dp-sequential-events/">PyPI</a> · <a href="https://github.com/martaajonees/dp-sequential-events/issues">Report an issue</a></sub>
</p>
