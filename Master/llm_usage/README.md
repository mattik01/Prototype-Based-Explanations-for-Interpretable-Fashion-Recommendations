# LLM-Usage Archive

Tangible record of how large language models (primarily Claude Code) assisted in producing this thesis. Maintained **continuously** throughout the work — not reconstructed at the end — as a matter of research transparency. At thesis finalization this folder is copied into the thesis as its `LLM-Usage/` appendix.

Full conversation logs are infeasible to include; instead this collects the durable, inspectable *scaffolding* through which the AI was directed and through which it maintained state.

## Contents (assembled over time)
- **`claude_md_versions/`** — numbered snapshots of `CLAUDE.md` (the project's standing instructions to Claude Code), preserving how the AI's guidance evolved. See the version log below.
- **Skill / command / MCP setup** — the Claude Code skills, slash commands, and MCP servers used consistently (e.g. `/scrutiny`, `/design-candidate`, `/protocol`, `/paper-assistant`, `/commit`) and their related artifacts.
- **AI memory & orchestration documents** — the persistent memory index and notes the assistant used to carry state across sessions.
- **Scrutiny-Protocol dossiers & ledgers** — the audit-and-repair working documents (`Master/docs/scrutiny/`). These are **preserved, not deleted**, when the Scrutiny phase ends, and a copy lands here.
- (Other AI-driven working artifacts as they accrue.)

## Conventions
- **CLAUDE.md versioning:** before any *substantive* change to `CLAUDE.md`, the current file is copied to `claude_md_versions/CLAUDE_vN.md` (next N; trivial typo/format fixes are skipped). Git history is the fine-grained backstop.
- **Preservation (at-risk only):** artifacts that would otherwise be lost to routine cleanup or refactoring — chiefly the Scrutiny dossiers at phase close, and superseded `CLAUDE.md` versions — are actively preserved here as they occur. Stable, persistent material (protocol vault, memory files, design-candidate docs, skill list) is not at risk and is simply gathered at finalization.

## CLAUDE.md version log
| Version | Date | Note |
|---|---|---|
| v1 | 2026-07-12 | Baseline captured just before adding the LLM-usage-archive rule to CLAUDE.md (already includes the same-day GPU-machine→cluster audit). |
| v2 | 2026-07-12 | Captured before reserving dc05's `feature_user_proto` in the ft_type table (dc05 Step 6 registration, fU design cycle). |
| v3 | 2026-07-13 | Captured before adding the LEO5 login-node smoke-test note (2× A30 on the login node, orders of magnitude faster than laptop CPU for local smokes; user directive during dc05 SC.3). |
| v4 | 2026-08-20 | Captured before adding the `lightfm_hist` ft_type row (dc05 S1 decoupled control, F-DC05-21 — dc05 SC.8 gate decision to run the full control suite). |
| v5 | 2026-09-09 | Captured before registering the dc08 arms (item-identity history rows inside the fU composition — knobs on `feature_user_proto`, five new model names). |
