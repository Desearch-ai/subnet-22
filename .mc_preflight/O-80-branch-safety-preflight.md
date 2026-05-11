# O-80 Branch Safety Preflight — SN22

**Task:** #1563 — `task/1563-o-80-sn22-branch-safety-preflight-and-reme`
**Date:** 2026-05-12
**Repo:** Desearch-ai/subnet-22
**Worktree:** `/tmp/mc-task-1563-subnet-22`

---

## Source State

| Field | Value |
|---|---|
| **Source branch** | `origin/objective/o-80-clean-sn22-objective-branch-remediation` |
| **Source commit** | `911c83bcbfc0c21b40691cb60ce1518d2a21b2aa` |
| **Local main vs origin/main** | Local main is clean; origin/main is identical to HEAD. No behind/ahead. |
| **Objective branch** | `objective/o-80-clean-sn22-objective-branch-remediation` |
| **Worktree branch** | `task/1563-o-80-sn22-branch-safety-preflight-and-reme` (integration, targets Objective branch) |

---

## Accepted O-74 Residual Gaps (inherited, not re-audited)

These gaps were accepted by the O-74 GAP review. They are the current state of origin/main and are NOT fixed by this task.

### Gap 1: Smart-Scrape Identity Drift
- **File:** `desearch/__init__.py` lines 43–44
- **Current values:** `ENTITY = "smart-scrape"` / `PROJECT_NAME = "smart-scrape-1.0"`
- **Problem:** Subnet 22's entity/project identity is "smart-scrape" but the product is Desearch. This is cosmetic (not a runtime error) but pollutes package introspection.
- **Evidence:** `python3 -c "import desearch; print(desearch.ENTITY, desearch.PROJECT_NAME)"` → `smart-scrape smart-scrape-1.0`
- **Remediation:** Future implementation task (not this preflight) should change these to `ENTITY = "desearch"` / `PROJECT_NAME = "subnet-22"` or appropriate Desearch branding. No entity name change may land to `main` without explicit Giga approval.

### Gap 2: Pytest Collection — uv-consumable Project Surface
- **File:** `pytest.ini` + `setup.py` (no `pyproject.toml`)
- **Problem:** `pytest --collect-only` produces 11 errors. The repo is not declared as a uv-consumable package with a proper test dependency surface. `setup.py` requires `pip install -e .` to function; uv cannot consume it as-is.
- **Current state:** 36 items collected with 11 errors. Errors are in test import resolution, not in production code.
- **Remediation:** Future implementation task (not this preflight) should add a `pyproject.toml` with `[project]` and `[project.optional-dependencies]` test declaration so `uv run pytest` works without pip-workaround. No pyproject.toml may land to `main` without explicit Giga approval.

---

## Branch Policy (Contract for Future Tasks)

1. **No direct push to `main`.** All implementation tasks must target `objective/o-80-clean-sn22-objective-branch-remediation`, NOT `main`.
2. **No task PR to `main` directly.** The integration branch (`task/*` branches) all merge into the Objective branch, not main. From Objective branch, a separate human-initiated PR merges to main with explicit Giga approval.
3. **Stale local main is not acceptance evidence.** "It passes tests on my local main" is not valid. All validation must be run against `origin/main` or the designated target branch.
4. **Untracked media / local-only files are not acceptance evidence.** Any artifact referenced in QA must be committed to the worktree or the Objective branch.
5. **No multi-repo checks.** This preflight contract applies ONLY to `Desearch-ai/subnet-22`. It does not authorize inspection of `cosmic-brain`, other Desearch repos, or workspace memory as acceptance evidence.

---

## What This Preflight Covers (Scope)

✅ Current origin/main commit hash recorded
✅ Objective branch name confirmed
✅ Local main sync status confirmed
✅ O-74 accepted gaps documented with file/line references
✅ Branch policy contract for future tasks
✅ Clear delineation of out-of-scope checks (other repos, stale local main)

## What This Preflight Does NOT Cover

- ❌ Any Cosmic Brain / other repo validation
- ❌ Fixing the two O-74 gaps (delegated to implementation tasks downstream)
- ❌ Landing to main (explicit Giga approval required separately)
- ❌ `pyproject.toml` creation (implementation task scope)
- ❌ `desearch/__init__.py` entity rename (implementation task scope)

---

## Validation Commands (for downstream tasks)

```bash
# Confirm branch and commit
git log origin/objective/o-80-clean-sn22-objective-branch-remediation --oneline -1
# Expected: 911c83b feat(task-487d8599): Fix SN22 package identity and pytest collection blockers from repo aud

# Confirm no direct main push intent
git show-branch origin/main origin/objective/o-80-clean-sn22-objective-branch-remediation

# Confirm entity drift
python3 -c "import desearch; print('ENTITY:', desearch.ENTITY, 'PROJECT:', desearch.PROJECT_NAME)"
# Currently: ENTITY: smart-scrape PROJECT: smart-scrape-1.0

# Confirm pytest collection errors (before fix)
python3 -m pytest --collect-only 2>&1 | grep -E "error|ERROR" | wc -l
# Currently: 11 errors
```

---

**This artifact is the contract. Future implementation tasks must not deviate from the branch policy above without explicit Giga approval.**
