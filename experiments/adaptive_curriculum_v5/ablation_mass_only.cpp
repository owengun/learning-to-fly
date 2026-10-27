#define ADAPTIVE_RUN_SUFFIX "_adaptive_v5_mass_only"
#define ADAPTIVE_DESCRIPTION "Adaptive curriculum v5 ablation: mass only, fixed reward"
#define ADAPTIVE_WIND_FACTORS 1.0, 1.0, 1.0, 1.0, 1.0
#define ADAPTIVE_REWARD_STAGES 0, 0, 0, 0, 0

#include "v5_precision_config.h"
#include "adaptive_curriculum_core.cpp"
