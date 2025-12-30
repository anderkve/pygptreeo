# Experimental Rationale

## Baseline Performance
**26.15% accuracy** (2092/8000 points within 2% error)

## Observations from Baseline
1. **Many convergence warnings**: L-BFGS optimizer failing to converge after limited iterations
2. **Hyperparameters hitting bounds**: Length scales reaching 1e5 bound, alpha reaching 1e4 bound
3. **Progressive improvement**: Accuracy improves from 6.9% to 26.15% as more data arrives
4. **Complex kernel**: Using AnisotropicRationalQuadratic + Matern combination

## Experiments Running

### Experiment 1: More Optimizer Restarts
**Change**: `n_restarts_optimizer: 3 → 5`
**Rationale**: Address convergence warnings by giving optimizer more chances to find better hyperparameters
**Expected impact**: Better kernel hyperparameters → improved predictions

### Experiment 2: More Frequent Retraining  
**Change**: `retrain_every_n_points: 200 → 100`
**Rationale**: Update GP models more frequently to adapt faster to data distribution
**Expected impact**: More responsive to local data patterns → potentially better early predictions

### Experiment 3: Product of Experts (PoE) Aggregation
**Change**: `aggregation: 'moe' → 'poe'`
**Rationale**: PoE can be more conservative and accurate when leaf GPs disagree
**Expected impact**: Better uncertainty estimates and potentially more accurate predictions in transition regions

## Additional Experiments to Consider (if time permits)
4. Wider hyperparameter bounds to prevent hitting limits
5. Simpler kernel (Matern only) to reduce optimization difficulty
6. More prediction leaves (max_n_pred_leaves: 5)
7. Hyperparameter inheritance (but requires disabling standard_scaling)
8. Point rejection to avoid redundant data
