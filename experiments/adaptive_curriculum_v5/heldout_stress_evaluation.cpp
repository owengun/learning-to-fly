// These factors are absent from the training curriculum and from the fixed
// force x3 / mass x1.2 validation task used for checkpoint selection.
#define STRESS_EVAL_FORCE_FACTORS 1.25, 1.75, 2.5, 3.5
#define STRESS_EVAL_MASS_FACTORS 1.025, 1.075, 1.15, 1.25

#include "stress_evaluation_core.cpp"
