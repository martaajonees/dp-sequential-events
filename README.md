# Differential Privacy in Sequential Event Logging
<p align="center">
<img src="https://badgen.net/badge/license/MIT/orange?icon=github" alt="license">
<img src="https://badgen.net/badge/language/Python/yellow" alt="language">
<img src="https://badgen.net/badge/build/passing/green?icon=githubactions" alt="build badge">
<img src="https://badgen.net/pypi/v/dp-sequential-events" alt="PyPI version">
<img src="https://img.shields.io/pypi/pyversions/dp-sequential-events?color=red" alt="Python version supported">
</p>

---

## ✨ Project Description
Sequential event logs often contain sensitive information. **dp-sequential-events** implements **differential privacy (DP)** techniques to anonymize sequential event logs while preserving statistical properties for analysis.

The pipeline follows these steps:

1. **DAFSA annotation** of event logs  
2. **Filtering** based on probabilistic risk measures  
3. **Differentially private case sampling**  
4. **Laplace noise injection** for timestamps  
5. **Reconstruction of anonymized timestamps**  
6. **Final privacy-preserving event log generation**

---

## 🗂 Repository Structure
```sh
dp-sequential-events
┣ 📂 src
┃ ┃
┃ ┗ 📂 dp_sequential_events
┃   ┣ 📂 main
┃   ┃ ┣ main.py
┃   ┃ ┣ annotated.py
┃   ┃ ┣ filtered.py
┃   ┃ ┗ case_sampling.py
┃   ┗ 📂 databases
┣ pyproject.toml
┗ requirements.txt
```

## 🚀 Online Execution

You can run the CLI in Google Colab or locally.
For Colab: [Open in Google Colab](https://colab.research.google.com/drive/17jejpDl4sX9L8885Pll4D_PJpxudtFL9?usp=sharing)

Install from PyPI ([see the project in PyPI](https://pypi.org/project/dp-sequential-events/)):
```
pip install dp-sequential-events
```
Run the CLI tool:
```
privseq
```
Run the pattern log functionality:
```
privseq-patterns
```

## 👩‍💻 Authors
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/martaajonees"><img src="https://avatars.githubusercontent.com/u/100365874?v=4?s=100" width="100px;" alt="Marta Jones"/><br /><sub><b>Marta Jones</b></sub></a><br /><a href="https://github.com/martaajonees/Local_Privacy/commits?author=martaajonees" title="Code">💻</a></td>
       <td align="center" valign="top" width="14.28%"><a href="https://github.com/ichi91"><img src="https://avatars.githubusercontent.com/u/41892183?v=4?s=100" width="100px;" alt="Anailys Hernandez" style="border-radius: 50%"/><br /><sub><b>Anailys Hernandez</b></sub></a><br /><a href="https://github.com/ichi91/Local_Privacy/commits?author=ichi91" title="Method Designer">💡</a></td>
    </tr>
     
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
