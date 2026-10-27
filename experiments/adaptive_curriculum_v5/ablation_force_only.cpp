#define ADAPTIVE_RUN_SUFFIX "_adaptive_v5_force_only"
#define ADAPTIVE_DESCRIPTION "Adaptive curriculum v5 ablation: force only, fixed reward"
#define ADAPTIVE_MASS_MAX_FACTORS 1.0, 1.0, 1.0, 1.0, 1.0
#define ADAPTIVE_REWARD_STAGES 0, 0, 0, 0, 0

#include "v5_precision_config.h"
#include "adaptive_curriculum_core.cpp"
