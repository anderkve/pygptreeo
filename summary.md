# pygptreeo: Gaussian Process Tree for Online Regression

## Overview

**pygptreeo** (Gaussian Process Tree for Online Regression) is a Python implementation of an adaptive, tree-based Gaussian Process regression framework designed for continual and online learning scenarios. The system implements a dynamically growing binary tree structure where each leaf node contains a local Gaussian Process (GP) regressor, enabling efficient handling of sequential data streams while maintaining probabilistic uncertainty quantification.

The implementation is inspired by the Deep Locally-Weighted Gaussian Processes (DLGP) framework and extends it with several novel features for improved memory efficiency, adaptive uncertainty calibration, and enhanced splitting strategies.

## Core Architecture

### Tree Structure

The **GPTree** class implements a binary tree that partitions the input space $\mathcal{X} \subseteq \mathbb{R}^d$ into overlapping regions. Each internal node defines a soft split based on a single dimension, while leaf nodes maintain local GP regressors trained on data within their respective regions.

**Key Properties:**
- **Dynamic Growth**: Nodes split when accumulating $N_{\text{bar}}$ training points
- **Overlapping Regions**: Controlled by parameter $\theta$, enabling smooth transitions between local models
- **Memory Efficiency**: Non-leaf nodes delete their data and GP models after splitting
- **Probabilistic Routing**: Data points are assigned to regions using soft, probabilistic splitting functions

### Node Architecture

Each **GPNode** manages:
- Local training data: $(X_i, y_i, \sigma_i)$ where $\sigma_i$ represents per-point uncertainty
- A scikit-learn Gaussian Process Regressor
- Split parameters: dimension $j$, position $s$, and overlap $o$
- Calibration statistics for adaptive uncertainty scaling
- Optional shared data from parent (for gradual splitting strategy)

## Mathematical Formulation

### Probabilistic Splitting Function

For a node with split dimension $j$ and position $s$, the probability of routing a point $\mathbf{x}$ to the right child is:

$$
p_{\text{right}}(\mathbf{x}) = \text{clip}\left(\frac{x_j - s}{o} + 0.5, \, 0, \, 1\right)
$$

where the overlap parameter is computed as:

$$
o = \theta \cdot \text{range}(X_{\text{train}, j})
$$

This creates a sigmoid-like transition region of width $o$ centered at $s$, allowing data points near the boundary to contribute to both child nodes.

### Prediction Aggregation

Given a test point $\mathbf{x}^*$, predictions are aggregated from multiple contributing leaf nodes using two strategies:

#### Mixture of Experts (MoE) - Default

The posterior predictive distribution combines predictions from $M$ contributing leaves weighted by their routing probabilities $\tilde{p}_m(\mathbf{x}^*)$:

$$
\mu(\mathbf{x}^*) = \sum_{m=1}^{M} \tilde{p}_m(\mathbf{x}^*) \mu_m(\mathbf{x}^*)
$$

$$
\sigma^2(\mathbf{x}^*) = \sum_{m=1}^{M} \tilde{p}_m(\mathbf{x}^*) \left(\sigma_m^2(\mathbf{x}^*) + \mu_m^2(\mathbf{x}^*)\right) - \mu^2(\mathbf{x}^*)
$$

where $\mu_m(\mathbf{x}^*)$ and $\sigma_m(\mathbf{x}^*)$ are the mean and standard deviation from leaf $m$.

#### Product of Experts (PoE)

Alternatively, predictions can be combined using a generalized product of experts formulation with weights $\beta_m = \tilde{p}_m(\mathbf{x}^*)$:

$$
\tau(\mathbf{x}^*) = \sum_{m=1}^{M} \beta_m \cdot \frac{1}{\sigma_m^2(\mathbf{x}^*)}
$$

$$
\mu_{\text{PoE}}(\mathbf{x}^*) = \frac{1}{\tau(\mathbf{x}^*)} \sum_{m=1}^{M} \beta_m \cdot \frac{\mu_m(\mathbf{x}^*)}{\sigma_m^2(\mathbf{x}^*)}
$$

$$
\sigma_{\text{PoE}}^2(\mathbf{x}^*) = \frac{1}{\tau(\mathbf{x}^*)}
$$

### Per-Point Uncertainty Integration

A key feature is the integration of per-point observation uncertainties $\sigma_i$ into the GP framework. The noise variance for each observation is set as:

$$
\alpha_i = \sigma_i^2
$$

This heteroscedastic noise model allows the GP to appropriately weight training points based on their individual reliability.

## Novel Features and Innovations

### 1. Adaptive Uncertainty Calibration

Each node maintains a calibration mechanism to ensure well-calibrated predictive uncertainties:

- Tracks recent prediction residuals: $r_i = |y_i - \hat{y}_i|$
- Computes empirical coverage of 1-sigma intervals
- Adaptively adjusts a scaling factor $s_{\sigma}$ to achieve target coverage of 68%
- Calibrated uncertainty: $\sigma_{\text{calibrated}} = s_{\sigma} \cdot \sigma_{\text{GP}}$

This addresses a common limitation of GP predictions where uncertainty estimates can be poorly calibrated, especially in tree-based partitioning scenarios.

### 2. Gradual Splitting Strategy

Unlike standard decision trees that perform hard splits, pygptreeo implements a **gradual splitting** mode:

- Upon splitting, both children initially receive copies of all parent data
- Shared data is stored separately and gradually removed as new points arrive
- Enables smooth transitions and prevents abrupt changes in predictions
- Reduces variance in local GP models during the transition period

The gradual removal is triggered when new points arrive, with the shared point furthest from the split boundary being removed first.

### 3. Point Rejection Mechanism

To improve memory efficiency, nodes can selectively reject well-predicted points:

**Rejection Criterion:**

$$
\text{reject} \iff \frac{|y - \hat{y}|}{|y|} < \epsilon_{\text{reject}}
$$

where $\epsilon_{\text{reject}}$ is a threshold (default: $10^{-4}$).

This is only activated after collecting a minimum number of points (default: 50) to ensure the GP is well-trained before becoming selective.

### 4. Point Merging via Inverse-Variance Weighting

When new points arrive very close to existing points (distance $< \delta_{\text{merge}}$, default: 0.01), they are merged using precision-weighted averaging:

$$
w_{\text{old}} = \frac{1}{\sigma_{\text{old}}^2}, \quad w_{\text{new}} = \frac{1}{\sigma_{\text{new}}^2}
$$

$$
\mathbf{x}_{\text{merged}} = \frac{w_{\text{old}} \mathbf{x}_{\text{old}} + w_{\text{new}} \mathbf{x}_{\text{new}}}{w_{\text{old}} + w_{\text{new}}}
$$

$$
y_{\text{merged}} = \frac{w_{\text{old}} y_{\text{old}} + w_{\text{new}} y_{\text{new}}}{w_{\text{old}} + w_{\text{new}}}
$$

$$
\sigma_{\text{merged}}^2 = \frac{1}{w_{\text{old}} + w_{\text{new}}}
$$

This reduces redundancy in the training set and improves computational efficiency.

### 5. Hyperparameter Inheritance

Child nodes can inherit their parent's optimized GP kernel hyperparameters:

- Parent's trained kernel parameters (length scales, amplitudes) are copied to children
- Provides warm-start for child GP optimization
- Accelerates convergence and improves local model quality
- Incompatible with standard scaling (which changes coordinate systems between parent/child)

### 6. Advanced Split Dimension Selection

Multiple criteria for choosing the split dimension $j$:

- **Max Spread**: $j = \arg\max_i \left(\max(X_{\cdot,i}) - \min(X_{\cdot,i})\right)$
- **Max Variance**: $j = \arg\max_i \text{Var}(X_{\cdot,i})$
- **Max Uncertainty**: Split on dimension where GP posterior variance is highest
- **Random**: Uniform random selection
- **Split Evaluation**: Evaluate multiple candidate splits and choose best (experimental)

The **split evaluation** mode tests multiple candidate splits by:
1. Partitioning data according to each candidate
2. Training temporary GPs on each partition
3. Computing validation performance
4. Selecting the split with best generalization

### 7. Custom Kernel Implementations

#### Anisotropic Rational Quadratic Kernel

Extends the standard Rational Quadratic kernel with per-dimension length scales:

$$
k(\mathbf{x}, \mathbf{x}') = \left(1 + \frac{d^2(\mathbf{x}, \mathbf{x}')}{2\alpha}\right)^{-\alpha}
$$

where the anisotropic squared distance is:

$$
d^2(\mathbf{x}, \mathbf{x}') = \sum_{i=1}^{d} \frac{(x_i - x'_i)^2}{\ell_i^2}
$$

The parameter $\alpha$ controls the scale mixture: small $\alpha$ behaves like a single RBF, while large $\alpha$ includes a wider range of length scales.

**Key Features:**
- Per-dimension length scales $\ell_i$ for anisotropic problems
- Full gradient implementation for hyperparameter optimization
- Suitable for problems with different characteristic scales per dimension

#### Additive Kernel with Configurable Interaction Depth

Implements additive decomposition for learning low-dimensional structure:

$$
k(\mathbf{x}, \mathbf{x}') = \sum_{\mathcal{S} \in \mathcal{T}} \prod_{i \in \mathcal{S}} k_i(x_i, x'_i)
$$

where $\mathcal{T}$ is the set of all interaction terms up to depth $D$.

**Example** (3D, depth=2):

$$
k(\mathbf{x}, \mathbf{x}') = k_1(x_1, x'_1) + k_2(x_2, x'_2) + k_3(x_3, x'_3) \\
+ k_1(x_1, x'_1) \cdot k_2(x_2, x'_2) \\
+ k_1(x_1, x'_1) \cdot k_3(x_3, x'_3) \\
+ k_2(x_2, x'_2) \cdot k_3(x_3, x'_3)
$$

**Configuration:**
- `interaction_depth=1`: Pure additive (main effects only)
- `interaction_depth=2`: Main effects + pairwise interactions
- `interaction_depth=3`: Up to 3-way interactions
- Base kernels: RBF or Matérn ($\nu=1.5$)

**Advantages:**
- Avoids curse of dimensionality for sparse, low-order functions
- Interpretable: identifies which dimensions/interactions matter
- Efficient for high-dimensional problems with additive structure
- Full gradient support for hyperparameter learning

### 8. Ensemble Method: GPForest

The **GPForest** class implements an ensemble of multiple GPTree instances:

**Training:** Multiple trees with different hyperparameters (varying $N_{\text{bar}}$, $\theta$, or kernel configurations)

**Prediction Aggregation:** Weighted by tree-specific uncertainties relative to prior:

$$
\alpha_i = \frac{1}{2}\left(\log \sigma_{\text{prior},i} - \log \sigma_i\right)
$$

$$
T_i = \frac{1}{\sigma_i^2}
$$

$$
\mu_{\text{forest}} = \frac{\sum_{i=1}^{N_{\text{trees}}} \frac{\alpha_i T_i}{\sum_j \alpha_j} \mu_i}{\sum_{i=1}^{N_{\text{trees}}} \frac{\alpha_i T_i}{\sum_j \alpha_j}}
$$

This provides improved robustness and stability compared to single-tree models.

## Implementation Details

### Online Update Algorithm

The `update_tree(x, y, σ)` method implements the core online learning loop:

1. **Route to Leaf**: Traverse tree using probabilistic routing from root to leaf
2. **Check Merging**: If point is near existing point (distance $< \delta_{\text{merge}}$), merge and return
3. **Check Rejection**: If point is well-predicted (relative error $< \epsilon_{\text{reject}}$), reject and return
4. **Store Point**: Add $(x, y, \sigma)$ to node's training set
5. **Update Statistics**: Register prediction performance for calibration
6. **Update Calibration**: Adjust $s_{\sigma}$ to maintain target coverage
7. **Retrain GP**: If buffer full ($n_{\text{buffer}} \geq n_{\text{retrain}}$) or node full
8. **Split Node**: If $n_{\text{points}} \geq N_{\text{bar}}$, create children and partition data
9. **Delete Parent Data**: Remove GP and data from parent to save memory

### Splitting Process

When a node reaches capacity:

1. **Select Split Dimension**: Using one of the configured criteria
2. **Compute Split Position**: Median, mean, or random within data range
3. **Create Children**: Two GPNode instances with inherited configuration
4. **Compute Overlap**: $o = \theta \cdot \text{range}(X_{\cdot,j})$
5. **Transfer Hyperparameters**: If enabled, copy optimized kernel to children
6. **Partition Data**:
   - Standard: Hard assignment based on split
   - Gradual: Both children get all data, marked as "shared"
7. **Delete Parent Resources**: Remove GP model and data from parent node

### Standard Scaling

Optional feature for coordinate normalization:

$$
X_{\text{scaled}} = \frac{X - \mu_X}{\sigma_X}, \quad y_{\text{scaled}} = \frac{y - \mu_y}{\sigma_y}
$$

Uncertainties are transformed accordingly:

$$
\sigma_{\text{scaled},i} = \frac{\sigma_i}{\sigma_y}
$$

Predictions are inverse-transformed before returning to user.

**Note:** Incompatible with hyperparameter inheritance since scaling changes the coordinate system between parent and child nodes.

## Performance Optimizations

### Recursive Leaf Collection

The `predict_recursive` mode uses tree traversal to collect only contributing leaves:

- Early stopping when cumulative probability $\geq 1.0$
- Skips subtrees with zero routing probability
- Optimal for large trees ($N_{\text{train}} \gg N_{\text{bar}}$)

### Loop-Based Prediction

The `predict_loop` mode iterates through all leaves:

- Computes marginal routing probability for each leaf
- Skips leaves with zero contribution
- Better for shallow trees with many test points

### Limiting Active Leaves

Parameter `max_n_pred_leaves` restricts number of contributing leaves:

- Select top-$k$ leaves by routing probability
- Renormalize probabilities: $\tilde{p}_m \leftarrow \tilde{p}_m / \sum_{i=1}^k \tilde{p}_i$
- Reduces computational cost for complex trees

## Use Cases and Applications

**Optimal Scenarios:**
- **Sequential/Online Learning**: Data arrives one point at a time
- **Continual Learning**: Model adapts to evolving data distributions
- **Heteroscedastic Noise**: Different reliability across observations
- **Large-Scale GP**: Computational tractability for datasets where standard GP inference is prohibitive
- **Adaptive Partitioning**: Function has varying complexity across input space
- **Memory Constraints**: Point rejection/merging controls memory footprint

**Limitations:**
- Not designed for batch learning (though `fit` method available)
- Binary splits may be inefficient for very high-dimensional problems
- Performance depends on appropriate $N_{\text{bar}}$ selection

## Technical Specifications

**Dependencies:**
- `numpy`: Numerical computations
- `scikit-learn`: Gaussian Process Regressors and kernels
- `binarytree`: Tree structure visualization
- `tqdm`: Progress bars
- `joblib`: Model serialization

**Key Parameters:**
- `Nbar`: Maximum points per leaf (default: 100)
- `theta`: Overlap parameter (default: 0.0001)
- `retrain_every_n_points`: GP retraining frequency (default: 100)
- `split_dimension_criteria`: Dimension selection method (default: 'max_spread')
- `splitting_strategy`: 'standard' or 'gradual' (default: 'standard')
- `use_calibrated_sigma`: Enable uncertainty calibration (default: True)
- `use_standard_scaling`: Enable data standardization (default: True)
- `use_hyperparameter_inheritance`: Enable kernel parameter inheritance (default: False)
- `enable_point_rejection`: Enable selective point rejection (default: False)
- `enable_point_merging`: Enable nearby point merging (default: False)
- `enable_split_evaluation`: Evaluate multiple split candidates (default: False)

## Research Contributions and Novel Aspects

1. **Heteroscedastic Online GP**: Full integration of per-point uncertainties in online tree-based GP framework

2. **Adaptive Calibration**: Automatic uncertainty calibration at node level ensures reliable confidence intervals

3. **Memory-Efficient Online Learning**: Point rejection and merging mechanisms enable long-running online learning with bounded memory

4. **Gradual Splitting**: Smooth transition during node splits reduces prediction variance and improves stability

5. **Hyperparameter Transfer**: Warm-starting child GPs with parent's optimized parameters accelerates training

6. **Anisotropic Kernels**: Custom implementation with proper gradients for optimization

7. **Additive Kernels**: Configurable interaction depth for learning sparse, low-dimensional structure in high-dimensional spaces

8. **Flexible Split Strategies**: Multiple criteria including uncertainty-based splitting for adaptive partitioning

## Code Quality and Testing

**Testing:**
- Unit tests for GPNode, GPTree, and standard scaling
- Integration tests for point rejection and uncertainty splitting
- Long-running stability tests
- Performance benchmarking suite

**Examples:**
- Basic usage demonstration
- Animated 2D visualization of online learning
- Comprehensive performance metrics tracking
- Standard benchmark functions (Eggholder, Himmelblau, Rosenbrock, Rastrigin, Levy)

**Documentation:**
- Comprehensive docstrings following NumPy/SciPy style
- README with installation and usage instructions
- Example scripts with detailed comments
- Mathematical notation in comments references DLGP framework

## License

GNU General Public License v3.0 (GPL-3.0) - Open source copyleft license ensuring derivative works remain open source.

## Relation to Literature

The implementation is inspired by the **Deep Locally-Weighted Gaussian Processes (DLGP)** framework, with several extensions:

- Original DLGP focuses on static tree construction
- pygptreeo emphasizes online/continual learning scenarios
- Novel memory management strategies (rejection, merging)
- Enhanced calibration mechanisms
- Extended kernel library
- Flexible splitting strategies beyond simple median splits

The codebase represents a production-ready implementation suitable for research in:
- Online Gaussian Process regression
- Continual learning with uncertainty quantification
- Adaptive function approximation
- Scalable Bayesian inference
- Transfer learning in sequential settings (via hyperparameter inheritance)
