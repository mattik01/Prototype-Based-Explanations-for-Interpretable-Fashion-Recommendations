Create a safety checkpoint commit before risky changes.

Steps:
1. Run `git status` to see all changes (staged, unstaged, untracked).
2. Stage ALL changes including untracked files (use specific filenames, never `git add -A`).
3. Create a single commit with prefix `checkpoint:` and a message describing the current stable state and what risky action is about to be attempted. Format:
   ```
   checkpoint: <stable state description> — before <upcoming risky thing>

   Co-Authored-By: Claude Haiku 4.5 <noreply@anthropic.com>
   ```
4. Run `git status` to verify clean state.
5. Do NOT push to remote.

These commits are identifiable via `git log --grep="checkpoint:"`.
