# marl-drag-reduction Guide Hub

This folder contains four coordinated documents derived from mining the marl-drag-reduction source tree using the topology-first workflow:

| File | Audience | Contents |
| --- | --- | --- |
| [`topology.md`](./topology.md) | Everyone (start here) | Structural map (Mermaid) + entity inventory + dependency edges. The backbone for the other docs. |
| [`developer_guide.md`](./developer_guide.md) | Framework developers / contributors | Architecture, module tours, extension SOPs, debugging tips. |
| [`user_guide.md`](./user_guide.md) | End users / applied engineers | Installation, core API, typical workflows, FAQ. |
| [`tutorial.md`](./tutorial.md) | Learners at any level | 10-chapter "from beginner to expert" walkthrough grounded in the repository's own configs and scripts. |

> Upstream project: https://github.com/KTH-FlowAI/MARL-drag-reduction-in-wall-bounded-flows
> Canonical paper: full citation in [README.md § Introduction](../../README.md#introduction).

> **Citation note:** the addendum driving this mining pass hypothesized this repo corresponds to Vignon et al., *Phys. Fluids* 35, 065146 (2023). The repo's own `README.md` and `setup.py` (`author="Luca Guastoni"`) instead point to the Guastoni et al. arXiv:2301.09889 paper above. This document set uses the repo's own citation; the Vignon et al. hypothesis is unconfirmed by anything in this repository and should be treated as a separate work.

## How these documents were produced

- Based on direct reading of `src/` (`configuration.py`, `run.py`, `evaluate.py`, `simson_marl.py`, `simsonutils3D.py`), `conf/README.md`, `data/README.md`, `runs/README.md`, and the top-level `README.md`.
- Every claim is grounded in a concrete source path; unresolved facts are marked with `> TODO(doc-miner): ...`.

## Reading order suggestions

1. **New users** — start with `user_guide.md` §1–§11, then Chapters 1–4 of `tutorial.md`.
2. **Applied engineers** with a specific task — jump to the matching chapter of `tutorial.md`.
3. **Contributors** — read `developer_guide.md` end to end; cross-reference `user_guide.md` when a user-facing API is discussed.
