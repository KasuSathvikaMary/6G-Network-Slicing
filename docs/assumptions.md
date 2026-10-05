# 6G Network Slicing Simulation Assumptions

The following assumptions and simplifications were made in the design and implementation of this simulation:

- **Time resolution**: 5-minute intervals (288 steps = 24 hours).
- **Resource model**: Physical Resource Blocks (PRBs) and Compute Units (CUs) are independently allocated.
- **Latency model**: M/M/1-inspired approximation; this is not a packet-level simulation.
- **Traffic**: Traffic is synthetic, combining a sinusoidal daily pattern with Bernoulli bursts.
- **QoS**: Quality of Service (QoS) evaluation uses threshold-based binary violations.
- **Power model**: Energy consumption is linear with respect to resource usage.
- **DRL**: The Deep Reinforcement Learning agent uses PPO with an MLP architecture and is trained on the default scenario.
- **Oracle**: The Oracle policy acts greedily with perfect knowledge of current demand; it does not solve for the global LP-optimal allocation over time.
- **SHAP**: Explainability uses a KernelExplainer with a sampled background of 100 points.
- **Drift**: Concept drift is detected using Population Stability Index (PSI) with fixed bins over a sliding window.
- **Audit**: The audit log uses a single-node hash chain and is NOT a true distributed ledger.
- **Realism**: All traffic is synthetic; real-network validation is left for future work.
- **DLT**: The Distributed Ledger Technology is simplified. Real consensus protocols have different latency and overhead characteristics.
- **Mobility**: No mobility, handovers, or channel fading effects are modeled.
- **Homogeneity**: Slices are assumed to be homogeneous within their respective types.
- **Interference**: There is no inter-slice interference modeled at the PHY layer.
