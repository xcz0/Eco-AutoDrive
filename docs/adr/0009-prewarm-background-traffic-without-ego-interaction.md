# Prewarm background traffic without ego interaction

**Status:** Accepted, not implemented

The target traffic initialization lets background traffic evolve while ego is absent from collision, blocking and traffic decisions, then introduces ego at the fixed evaluation anchor with continuous history and stable object IDs. This preserves real history while avoiding the initialization disturbance caused by a stationary ego; implementation is tracked in GitHub Issue #2.
