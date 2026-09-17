# Review findings

The reviewer appends a section per reviewed commit range. Each finding has a
severity (crash / correctness / concurrency / test-gap / style), a
`file:line`, a concrete failure scenario, and a status. The worker sets
status to `fixed <hash>` or `wontfix <reason>`; the reviewer may reopen.
Findings with status `open` block new plan items.

(no findings yet)
