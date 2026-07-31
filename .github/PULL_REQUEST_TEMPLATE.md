## What and why

<!-- Normal description for reviewers. -->

## Impact

<!--
Fill the block below. It is the only place "what does a customer notice, and
why did we do this" is ever cheap and accurate, because you know it right now
and a diff cannot answer it later.

`significant: false` is a complete answer and needs nothing else. Use it for
internal renames, behaviour-preserving refactors, formatting, editorial doc
edits, tests, dependency bumps, build changes, and internal logging.

Use `significant: true` when someone outside the team could notice without
reading the code: a new or removed endpoint, field, screen or permission; a
changed request or response shape; the same call returning something different;
changed limits, quotas, timeouts or price; accuracy or latency changed enough to
alter a published claim; changed supported regions, integrations or client
versions; changed authentication, retention or what data is collected.

It is not a size test. A one-line rate-limit change counts. A thousand-line
refactor that preserves every behaviour does not.
-->

<!-- company-os:impact -->
```yaml
significant: false
```

<!-- When significant, use this shape instead:

<!-- company-os:impact -->
```yaml
significant: true
category: surface-added   # surface-added | surface-removed | shape-changed | behaviour-changed | limits-changed | quality-changed | availability-changed | security-or-privacy-changed
user_impact: >-
  What a customer can now do, or can no longer do, in their words.
why: >-
  The problem this removes. Not the code.
evidence: https://...     # docs, changelog, or a screenshot
```
-->
