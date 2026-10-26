# Adaptive curriculum v3

V3 adds a dual-objective gate to the staged v2 curriculum.

Every 25k interactions it evaluates:

- nominal validation: 10 episodes for hover precision;
- severe stress validation: 30 episodes at force x3 and mass x1.2.

A stage promotion requires three consecutive passes on both validations.
Rollback requires two consecutive retention failures and also checks the severe
stress condition, so a policy cannot retain a higher stage merely by performing
well in the nominal environment.

The run lasts 400k interactions and retains 300k and 400k checkpoints. Build
target: `adaptive_curriculum_v3_training`.

