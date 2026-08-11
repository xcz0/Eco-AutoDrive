# Require explicit programmatic speed limits

Treat a programmatic lane speed as an explicit experiment condition and replace only MetaDrive's exact unset sentinel while preserving legitimate lane limits. This avoids silently changing map semantics or letting an unset value enter the model as an extreme valid speed.
