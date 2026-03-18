# Codebase Deep Understanding Workflow

## Process
Sessions 01–03 use a **Q&A format**: Claude prepares questions, user answers, Claude evaluates.
Sessions 04–10 use a **guided walkthrough format**: Claude walks you through the code, pointing at specific lines, explaining concepts, with "pause and check" moments and self-assessment checklists.

Both formats achieve the same depth of understanding — the walkthrough format is more interactive and keeps you in the code while learning.

## Instructions for Walkthroughs (Sessions 04–10)
- Open the referenced source file(s) side-by-side
- Follow along line by line — the walkthrough references specific line numbers
- When you see "**Pause and check**", stop and think about the question before reading the answer
- At the end, use the "Check Your Understanding" checklist to verify comprehension
- If any checklist item is unclear, re-read that section or ask Claude for help

## File naming
`0X_<topic>.md`

## Learning Path (follows data flow, bottom-up to prototypes)

| # | Topic | Key Files | Format |
|---|-------|-----------|--------|
| 01 | Data Layer | `protomf_dataset.py`, a splitter | Q&A |
| 02 | Config & Entry | `consts.py`, `hyper_params.py`, `start.py` | Q&A |
| 03 | Orchestration | `experiment_helper.py` | Q&A |
| 04 | Core Model | `rec_sys.py` | Walkthrough |
| 05 | Training & Evaluation | `trainer.py`, `tester.py`, `eval.py` | Walkthrough |
| 06 | Feature Extractors (Base) | `feature_extractors.py` (Embedding, EmbeddingW, ACF) | Walkthrough |
| 07 | Feature Extractors (Prototypes) | `feature_extractors.py` (PrototypeEmbedding) | Walkthrough |
| 08 | Factory & Assembly | `feature_extractor_factories.py` | Walkthrough |
| 09 | Explanations | `explanations_utils.py` | Walkthrough |
| 10 | Integration Exercise | Full end-to-end trace | Walkthrough |

## Progress
- 01_data_layer.md ✅
- 02_config_entry.md ✅
- 03_orchestration.md ⬜ (partially answered)
- 04_core_model.md ⬜
- 05_training_evaluation.md ⬜
- 06_extractors_base.md ⬜
- 07_extractors_prototypes.md ⬜
- 08_factory_assembly.md ⬜
- 09_explanations.md ⬜
- 10_integration_exercise.md ⬜
