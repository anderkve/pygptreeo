# PyGPTreeo Performance Improvement Experiments

## Baseline Configuration
- Kernel: ConstantKernel * (AnisotropicRationalQuadratic + Matern)
- Nbar: 200
- theta: 1e-4
- retrain_step: 200
- n_restarts_optimizer: 3
- splitting_strategy: gradual
- split_dimension_criteria: max_uncertainty
- max_n_pred_leaves: 3
- aggregation: moe
- enable_split_evaluation: True (n_split_candidates=4)
- use_standard_scaling: True
- use_hyperparameter_inheritance: False

## Baseline Results
- **Accuracy: 26.15%** (2092/8000 points within 2%)
- Progress: 6.9% @ 1k → 26.15% @ 8k points

## Experiments to Try

### Experiment 1: More optimizer restarts + wider bounds
- n_restarts_optimizer: 3 → 5
- length_scale_bounds: (1e-5, 1e5) → (1e-6, 1e6)
- alpha_bounds: (1e-4, 1e4) → (1e-5, 1e5)

### Experiment 2: More frequent retraining
- retrain_step: 200 → 100

### Experiment 3: Product of Experts aggregation
- aggregation: 'moe' → 'poe'

### Experiment 4: More prediction leaves
- max_n_pred_leaves: 3 → 5

### Experiment 5: Simpler kernel (just Matern)
- Remove AnisotropicRationalQuadratic, use only Matern

### Experiment 6: Hyperparameter inheritance
- use_hyperparameter_inheritance: True
- use_standard_scaling: False (conflicts with inheritance)

## Results

| Experiment | Accuracy | Change |
|------------|----------|--------|
| Baseline   | 26.15%   | -      |
|            |          |        |
