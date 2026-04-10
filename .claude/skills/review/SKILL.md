---
name: review
description: 3-agent parallel code review (security, performance, test coverage)
---
Run `git diff main...HEAD` to get all changes on this branch. Launch 3 review agents in parallel:
1. Security: CORS, input validation, path traversal, injection. Rate Critical/High/Medium/Low.
2. Performance: memory, concurrency, hot-path bloat, unnecessary I/O.
3. Test coverage: untested paths, missing edge cases, top 5 tests to write.
Aggregate findings. Fix Critical/High directly. Report Medium/Low.
