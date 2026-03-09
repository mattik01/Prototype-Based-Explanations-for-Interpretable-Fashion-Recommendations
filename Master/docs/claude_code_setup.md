# Claude Code Setup Requirements

This document lists the plugins and tools Claude Code expects to have installed for this project. These are **per-machine** settings and must be reconfigured on each new installation.

## Required Plugins

| Plugin | Purpose | Install command |
|--------|---------|-----------------|
| `context7` | Live documentation lookup for libraries (PyTorch, Ray, wandb, etc.) | `claude plugin install context7@claude-plugins-official` |
| `pyright-lsp` | Python type checking and static analysis | `claude plugin install pyright-lsp@claude-plugins-official` |
| `code-simplifier` | Code review and simplification via `/simplify` | `claude plugin install code-simplifier@claude-plugins-official` |
| `claude-md-management` | CLAUDE.md auditing, maintenance, and improvement | `claude plugin install claude-md-management@claude-plugins-official` |

## Other Configuration

The following are also configured in `~/.claude/settings.json` but are optional:

- **Status line:** `ccstatusline` (custom status bar via `npx -y ccstatusline@latest`)
- **Notification hook:** Plays a sound on task completion (`paplay`)

## Verification

After installing, run `claude plugin list` to confirm all plugins are active. They should also appear in the system prompt when starting a session.

## Notes

- Project-level instructions in `CLAUDE.md` and `~/.claude/` memory files travel with the repo or user profile respectively — only plugin/MCP configs need manual setup.
- The conda environment (`protomf`) must also be set up separately — see `CLAUDE.md` for details.
