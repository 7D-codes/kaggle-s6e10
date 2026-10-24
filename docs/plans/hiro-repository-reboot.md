# Collaboration Reboot and Source-Only Onboarding

- Agent: Hiro / Hermes / gpt-6.1-sol.
- Status: COMPLETE; training remains PAUSED.
- Branch: main.
- Strategy: make collaboration the entry point; share identity, strategy, ownership and per-task plans; preserve scientific history; verify software tests without private local artifacts.
- Evidence: source-only regression reproduced FileNotFoundError because test_route_confirmation.py copied an ignored private manifest.
- Owned: AGENTS.md, README.md, docs/, test_route_confirmation.py. No model implementation changes.
- Boundaries: training paused; no jobs, submissions, checkpoint downloads or deletion of scientific evidence.

## Progress
- [x] Pull and publish shared rules, strategy and experiment history.
- [x] Reproduce source-only manifest dependency (RED).
- [x] Replace only that test dependency with an explicitly synthetic fixture; production manifests and validation unchanged.
- [x] Make README collaboration-only; preserve project explanation in docs/PROJECT.md and add per-agent plans.
- [x] Full source-only suite:48passed,66existing warnings,13.70s; post-cleanup targeted smoke:1passed,2.55s.
- [x] Removed root Python/pytest caches and completed task-owned synthetic fixtures only. All datasets/models/predictions/states/manifests/evidence retained.
- [x] Prepared final handoff for the verified main-branch push; commit history provides its identity.

## Verification
Initial source-only targeted check:1expected failure for absent ignored inference manifest. Corrected full source-only suite passed48tests without ignored data/results or Kaggle access. Post-cleanup targeted regression passed. All local Markdown links resolve. A passing workstation suite alone did not establish teammate onboarding.

## Handoff
Ownership released; no files remain reserved. Cleanup removed4091062logical bytes; observed free space rose354861056→360497152bytes during that measurement. This is repository housekeeping, not a root-cause fix for host storage pressure. Full scientific archive is retained.

No teammate identity invented. Each teammate adds their own registry entry/plan after pulling. Private artifact access is separate from public Git access. Requested collaborator invite remains unresolved: supplied email did not resolve to a public GitHub username; no invitation sent.
