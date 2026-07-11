# Scrutiny Findings Ledger

> Maintained by the Scrutiny Protocol (`Master/docs/scrutiny_protocol.md`).
> One entry per non-trivial finding; trivial fixes get a one-line entry.
> **This ledger owns the WHY** (finding, severity, disposition, resolving
> commit); `Master/docs/modifications_log.md` owns the file-centric WHAT,
> cross-referenced by finding ID. Rationale is written once, here.
>
> **Schema per entry:**
> - ID: `F-S0-NN` (foundation) or `F-DC0N-NN` (candidate-scoped), sequential
> - applies-to: `shared` | `dc01`..`dc04` — `shared` findings are re-checked
>   at every later candidate's SC.1a
> - severity: `trivial` | `minor` | `major` | `blocker`
> - status: `open` → `fixed(<commit>)` | `wontfix` | `superseded(<by-ID>)` | `stale`
> - body: the finding, then `**Proposal:**` (the bundled fix), then
>   `**Disposition:**` (gate decision + date)
>
> Entry template:
>
> ```
> ### F-DC0N-NN [severity] [applies-to] [status]
> <one-sentence finding>
> <evidence / where>
> **Proposal:** <the bundled fix, implementation-ready>
> **Disposition:** <gate decision, date, resolving commit if fixed>
> ```

## Foundation (F-S0-…)

_(none yet)_

## dc01 (F-DC01-…)

_(none yet)_

## dc02 (F-DC02-…)

_(none yet)_

## dc03 (F-DC03-…)

_(none yet)_

## dc04 (F-DC04-…)

_(none yet)_
