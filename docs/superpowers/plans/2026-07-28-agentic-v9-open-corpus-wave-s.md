# Wave S implementation plan

1. Add tests proving official v9 uses open-corpus scope while preserving
   `expected_sources`.
2. Add a repository query for the evaluated user's owned document IDs and test
   its user filter.
3. Extend v9 admission/runtime with an explicit open-corpus path while retaining
   explicit-source diagnostic behavior.
4. Add a distinct execution profile and trace scope policy.
5. Run focused tests, broader evaluation tests, compile/lint checks, inspect the
   diff, and commit locally without pushing.
