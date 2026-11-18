[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.8335291-blue)](https://doi.org/10.5281/zenodo.8335291) [![codecov](https://codecov.io/gh/ukaea/process/graph/badge.svg?token=F94XDNUIX0)](https://codecov.io/gh/ukaea/process)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/ukaea/PROCESS/HEAD?urlpath=%2Fdoc%2Ftree%2Fexamples%2F)
# PROCESS

**Here are the [PROCESS docs](https://ukaea.github.io/PROCESS/).**

PROCESS is the reactor systems code at the [UK Atomic Energy Authority](https://www.ukaea.org/) (UKAEA). More information can be found on the PROCESS [webpage](https://www.ukaea.org/service/process/).

PROCESS was originally a Fortran code, but is now a pure-Python command line program. PROCESS is still undergoing a significant restructure and, as such, **PROCESS version 3 is unstable and does not guarantee backward compatibility**. PROCESS version 4 will be the first major version to enforce backward-compatible API changes and will be released following a refactor of the data structure.

---

## 本仓库改造说明 / Local remodeling

> **本目录是本地 fork，非上游官方树。** 基线为同级目录 [`process-original`](../process-original)（PROCESS v3.4.2）；本仓库在其上持续改造，用于 **p-¹¹B（质子–硼-11）** 无中子先进燃料研究。改造仍在进行中（WIP），请勿当作已合入 UKAEA/PROCESS 的功能。

### 目的

在 PROCESS v3.4.2 上增加 **p-¹¹B 聚变反应通道**：计算反应率与聚变功率，并将每个反应产生的 **3 个 α** 计入 α 功率/产率（无中子功率）。原版仅支持 D-T、D-D、D-³He。

### 相对 `process-original` 的差异

| | `process-original` | `process-pB`（本仓库） |
|---|---|---|
| 物理 | 上游 v3.4.2 | 上游 + p-¹¹B 反应与 α 贡献 |
| Windows 补丁 | `process_output.py`：`close_idempotence_files()` 先 close 再 unlink | 相同补丁 |
| 用途 | 对照基线、复现原版结果 | p-¹¹B / 无中子先进燃料研究 |

### 主要改动文件

| 文件 | 改动 |
|---|---|
| `process/core/constants.py` | `M_BORON11_AMU`、`P_B11_ENERGY`（8.683 MeV）、`P_B11_ALPHA_COUNT=3`、无中子份额等常量 |
| `process/core/input.py` | 输入 `f_nd_boron11_electrons` |
| `process/data_structure/physics_variables.py` | `nd_plasma_boron11_vol_avg`、`fusden_plasma_p11b`、`fusden_plasma_p11b_alpha`、`pden_plasma_p11b_mw`、`f_nd_boron11_electrons` 等 |
| `process/models/physics/fusion_reactions.py` | `p11b_reaction()` / `p11b_reactivity()`，接入反应循环；α 计入功率/产率，无中子功率 |
| `process/models/physics/physics.py` | 硼-11 参与准中性、Z_eff、离子密度/质量；新变量输出 |
| `process/models/costs/costs.py` | `cdrlife_cal==0`（无中子 / 包层寿命为 0）时的除零保护 |
| `process/core/process_output.py` | 与 original 相同的 Windows 文件句柄补丁 |
| `tests/unit/test_p11b_reaction.py` | p-¹¹B 单元测试 |
| `documentation/source/physics-models/fusion_reactions/p11b_reactions.md` | 反应模型说明 |

完整文件级 diff 见工作区 [`docs/pB_changes.diff`](../docs/pB_changes.diff)；部署与补丁清单见 [`docs/PROCESS本地部署说明.md`](../docs/PROCESS本地部署说明.md) §7.4。

### 新增输入 / 输出

**输入**

| 变量 | 含义 | 默认 |
|---|---|---|
| `f_plasma_fuel_protium` | 质子占燃料离子分数（5 组分归一） | 0 |
| `f_plasma_fuel_boron11` | 硼-11 占燃料离子分数（5 组分归一） | 0 |
| `f_nd_boron11_electrons` | 遗留：硼-11 密度 / 电子密度（与燃料分数互斥） | 0 |
| `f_nd_protium_electrons` | 遗留：质子密度 / 电子密度（与燃料分数互斥） | 0 |

燃料归一：`f_D + f_T + f_He3 + f_p + f_B = 1`。纯 p-¹¹B 用例如 `f_p=5/6, f_B=1/6`。

**输出（节选）**

- `pden_plasma_p11b_mw` / `p_plasma_p11b_mw`：p-¹¹B 聚变功率密度 / 功率
- `fusden_plasma_p11b`：反应率密度
- `fusden_plasma_p11b_alpha`：α 产生率密度（= 3 × 反应率）
- `nd_plasma_boron11_vol_avg`：硼-11 体平均密度

p-¹¹B 功率计入 α 通道（`pden_plasma_alpha_mw` 等）；不放入非 α 带电粒子通道，避免重复计数。

### 已知限制（WIP）

1. **硼线辐射 / 显式轫致未纳入**——成分记账已通，但硼轫致仍缺（见主 TODO [D1]）；纯 pB 功率平衡尚不可作燃烧判据。
2. **热 α 灰密度仍为输入**（`f_nd_alpha_thermal_electron`），未从 p-¹¹B 源项自洽求解。
3. α 在离子/电子间的能量分配仍沿用 D-T α 模型（对 p-¹¹B α 为近似）。
4. 氚增殖、中子壁负载、屏蔽等仍按 D-T 假设。
5. 改造进行中，**非上游**；双温 / 温度量程见主 TODO [A1][A2]。

### 验证（2026-10-20）

- D-T `large_tokamak_eval` 与 `process-original` **逐位一致**（净电功率 `385.30194778577436 MW`）。
- p-¹¹B 算例：α 产率 = 3 × 反应率。
- `pytest tests/unit/test_p11b_reaction.py` → **11 passed**。

### 如何运行（相对 `process-original`）

本仓库源码配合工作区 `venvs\pB`；原版用 `venvs\original`。推荐用同级启动器，避免直接调用已失效的 `process.exe`：

```powershell
cd C:\Users\82538\DSHwork\PROCESS
.\tools\run.cmd pB       runs\eval-pB\large_tokamak_eval.IN.DAT --no-pause
.\tools\run.cmd original runs\eval-original\large_tokamak_eval.IN.DAT --no-pause
```

也可双击 `tools\run-pB.cmd` / `tools\run-original.cmd` 后拖入 `IN.DAT`。更多说明见工作区 [`README.md`](../README.md) 与 [`docs/PROCESS本地部署说明.md`](../docs/PROCESS本地部署说明.md)。

---

![Blender_and_output](./documentation/source/images/README_image.PNG)
<center>Overview of some outputs for a DEMO-like reactor.</center>


## Getting Started
Please see the [installation guide](https://ukaea.github.io/PROCESS/installation/installation/) and the [usage guide](https://ukaea.github.io/PROCESS/usage/running-process/). Once installed, take a look at the [examples page](https://ukaea.github.io/PROCESS/usage/examples/) for examples of how PROCESS can be run, and its results visualised. 

If you wish to run the examples before installing, you can click on the `binder` badge at the top of the README, or 
can click [here](https://mybinder.org/v2/gh/ukaea/PROCESS/HEAD?urlpath=%2Fdoc%2Ftree%2Fexamples%2F). Once loaded, you will be able to run the PROCESS examples in your web browser.


## Documentation
To read about how the code works and the modules in it see the [documentation](https://ukaea.github.io/PROCESS/).

## Tracking and Testing
Process uses a mixture of tracking and testing to ensure code reliability. Tests are run on each branch and ensure the output of key functions are correct. Tracking, on the other hand, shows graphs of changes to variables over time, and what commit caused the change.

* Our tracker can be found here: https://ukaea.github.io/PROCESS/tracking.html
* Documentation on testing can be found here: https://ukaea.github.io/PROCESS/development/testing/

## Development
Please see the [CONTRIBUTING.md](https://github.com/ukaea/PROCESS/blob/main/CONTRIBUTING.md) for guidance on how to contribute to PROCESS. Further information is found in the development section of the [documentation](https://ukaea.github.io/PROCESS/development/git-usage/).

## Publications
A list of publications using PROCESS is given [here](https://ukaea.github.io/PROCESS/publications), including two papers outlining the physics and engineering models in PROCESS.

## Citing PROCESS
If you use PROCESS in your work, please cite it using the "Cite this repository" link in the "About" section of the repository. This will cite the latest version of PROCESS, if you are using a different release, please find the appropriate DOI on [PROCESS' Zenodo page](https://doi.org/10.5281/zenodo.8335291). To ensure reproducible research, we recommend you run an [official release of PROCESS](https://github.com/ukaea/PROCESS/releases) by checking out the appropriate git tag.

## Contact and Collaboration
If you would like to get in touch with the PROCESS group for a more detailed discussion, please use the contact form on the [PROCESS webpage](https://www.ukaea.org/service/process/) to make contact with the UKAEA maintainers of PROCESS. 

We can, at our discretion, provide some free modeller time without a formal collaboration agreement. This can help you towards converging your problem, and can be a good starting point for further collaboration; please contact us if you are interested. Alternatively, you can make an issue or discussion in line with our [contribution guide](CONTRIBUTING.md).