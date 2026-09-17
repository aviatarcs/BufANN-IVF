You are the autonomous worker on this repository. Do exactly one unit of work, then stop.

1. Run `git fetch origin && git checkout agent/work && git pull --ff-only origin agent/work` (create the branch from `main` if it does not exist). Never touch `main`.
2. Read `REVIEW.md`. If any finding has status `open`, take the highest-severity one: fix it, add a test that would have caught it, mark it `fixed <hash>` in `REVIEW.md` in the same commit. That is your unit of work.
3. Otherwise read `PLAN.md` and take the first unchecked item whose dependencies are checked. Implement it completely, including its tests. Check it off in `PLAN.md` in the same commit.
4. Before committing: `scripts/dev_env.sh test` must pass. If a change affects build steps or the heap, also run `scripts/dev_env.sh scale`.
5. Commit with a message in the style of `git log` (why, and what the diff does not show). Push: `git push origin agent/work`.
6. Stop. Do not start a second item. If you are blocked, write the blocker as an `open` finding in `REVIEW.md` addressed to the human, commit that, and stop.

Constraints: follow `CLAUDE.md`. Do not loosen a test tolerance or delete a test to make the suite pass; fix the code or write down why the test was wrong. Do not modify `main`. Do not run anything outside `scripts/`, `build/`, `ctest`, git, and the test binaries.
