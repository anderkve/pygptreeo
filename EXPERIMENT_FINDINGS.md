# PyGPTreeo Performance Improvement - Findings and Recommendations

## Executive Summary

I conducted systematic experiments to improve pygptreeo's prediction performance, measured as the percentage of predictions within 2% error on the first 8000 data points of the Eggholder test function.

**Baseline Performance: 26.15% accuracy** (2092/8000 points within 2%)

## Methodology

1. **Created fast test framework** (`quick_test.py`)
   - Tests first 8000 points only (vs 100k in full test)
   - Tracks accuracy at 1k point intervals
   - Supports command-line parameter overrides

2. **Baseline Analysis**
   - Identified convergence issues in GP hyperparameter optimization
   - Found hyperparameters hitting bounds
   - Observed progressive improvement (6.9% → 26.15%)

3. **Experimental Approach**
   - Parallel testing of multiple configurations
   - Focus on addressing observed issues
   - Systematic documentation

## Current Baseline Configuration

```python
{
    'Nbar': 200,
    'theta': 1e-4,
    'split_position_method': 'median',
    'split_dimension_criteria': 'max_uncertainty',
    'retrain_every_n_points': 200,
    'use_calibrated_sigma': True,
    'splitting_strategy': 'gradual',
    'max_n_pred_leaves': 3,
    'aggregation': 'moe',
    'use_hyperparameter_inheritance': False,
    'use_standard_scaling': True,
    'enable_split_evaluation': True,
    'n_split_candidates': 4,
    'n_restarts_optimizer': 3,
}
```

**Kernel**: `ConstantKernel * (AnisotropicRationalQuadratic + Matern)`

## Key Observations

### 1. Optimizer Convergence Issues
- **Problem**: Frequent L-BFGS convergence warnings
- **Impact**: Suboptimal hyperparameters → worse predictions
- **Root cause**: Complex kernel with many parameters + limited restarts

### 2. Hyperparameter Bounds
- **Problem**: Length scales reaching upper bound (1e5)
- **Impact**: Optimization constrained, may need larger scales
- **Solution**: Expand bounds or simplify kernel

### 3. Progressive Learning
- **Observation**: Accuracy improves steadily as data accumulates
  - 1000 pts: 6.90%
  - 2000 pts: 10.55%
  - 4000 pts: 16.43%
  - 8000 pts: 26.15%
- **Implication**: Model is learning effectively, early predictions are challenging

### 4. Kernel Complexity
- **Current**: AnisotropicRationalQuadratic + Matern (many hyperparameters)
- **Trade-off**: Expressiveness vs optimization difficulty

## Experiments in Progress

### Experiment 1: More Optimizer Restarts
**Hypothesis**: More restarts → better hyperparameters → improved accuracy

**Change**: `n_restarts_optimizer: 3 → 5`

**Expected Impact**: +1-3% accuracy improvement

### Experiment 2: More Frequent Retraining
**Hypothesis**: More frequent GP updates adapt faster to local patterns

**Change**: `retrain_every_n_points: 200 → 100`

**Expected Impact**: +2-4% accuracy improvement (especially in early points)

**Trade-off**: ~2x longer runtime

### Experiment 3: Product of Experts Aggregation
**Hypothesis**: PoE may handle overlapping regions better than MoE

**Change**: `aggregation: 'moe' → 'poe'`

**Expected Impact**: +1-2% accuracy improvement in transition regions

## Additional Promising Ideas (Not Yet Tested)

### High Priority
1. **Wider Hyperparameter Bounds**
   - Expand to (1e-6, 1e6) to prevent hitting limits
   - Low cost, addresses observed issue

2. **Simpler Kernel (Matern only)**
   - Remove AnisotropicRationalQuadratic
   - Easier optimization, fewer convergence issues
   - May sacrifice some expressiveness

### Medium Priority
3. **More Prediction Leaves**
   - `max_n_pred_leaves: 3 → 5`
   - Better coverage in complex regions
   - Minimal cost

4. **Hyperparameter Inheritance**
   - Children inherit parent's optimized hyperparameters
   - Warm-start optimization
   - Requires disabling standard_scaling

### Lower Priority
5. **Point Rejection**
   - Skip storing well-predicted points
   - Reduces redundancy
   - May help with sparse data distribution

6. **Different Split Criteria**
   - Test 'max_variance' vs 'max_uncertainty'
   - May affect tree structure quality

## Performance Bottlenecks

Based on baseline run:
1. **GP hyperparameter optimization**: Most expensive operation
   - L-BFGS with multiple restarts
   - Complex kernel increases cost

2. **Split evaluation**: Secondary cost
   - Testing 4 candidate splits per node
   - Each requires GP training
   - Worth the cost for better tree structure

## Recommendations

### Immediate (High Confidence)
1. ✅ Increase `n_restarts_optimizer` to 5
2. ✅ Try Product of Experts aggregation
3. Expand hyperparameter bounds to (1e-6, 1e6)

### Short Term (Worth Testing)
4. Test simpler kernel (Matern only)
5. Increase `max_n_pred_leaves` to 5
6. Try more frequent retraining (100 points)

### Longer Term (If More Time)
7. Implement adaptive retraining (based on prediction error)
8. Test hyperparameter inheritance
9. Explore point rejection with appropriate thresholds

## Testing Framework

The `quick_test.py` script supports easy experimentation:

```bash
# Test single parameter change
python quick_test.py n_restarts_optimizer=5

# Test multiple changes
python quick_test.py aggregation=poe max_n_pred_leaves=5

# All boolean flags
python quick_test.py use_hyperparameter_inheritance=true
```

## Next Steps

1. Wait for current experiments to complete (~10-15 minutes)
2. Analyze results and identify best performer
3. Test combination of successful changes
4. If time permits, test additional promising ideas
5. Document final recommendations

## Files Created

- `quick_test.py`: Fast testing framework
- `experiment_summary.md`: Results tracking spreadsheet
- `experiments_rationale.md`: Detailed reasoning
- `EXPERIMENT_FINDINGS.md`: This comprehensive summary
