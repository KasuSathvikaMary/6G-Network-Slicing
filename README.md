# 🔬 6G Network Slicing with AI-Driven Dynamic Resource Management

[![CI](https://github.com/KasuSathvikaMary/6G-Network-Slicing/actions/workflows/ci.yml/badge.svg)](https://github.com/KasuSathvikaMary/6G-Network-Slicing/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)

A **production-ready, experimentally validated** platform for intelligent 6G network slice orchestration using **Deep Reinforcement Learning (DRL)**, **Explainable AI (XAI)**, and **energy-aware resource management**.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [How It Works](#-how-it-works)
- [Research Contributions](#-research-contributions)
- [Features](#-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Running Experiments](#-running-experiments)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Docker Deployment](#-docker-deployment)
- [Testing](#-testing)
- [Project Structure](#-project-structure)
- [Performance Results](#-performance-results)
- [Citation](#-citation)
- [License](#-license)

---

## 🌐 Overview

This platform addresses the critical challenge of **dynamic resource orchestration** in 6G networks through an integrated approach combining:

- **Multi-Agent Deep Reinforcement Learning** for autonomous, real-time resource allocation
- **Explainable AI** for transparent, auditable decision-making
- **Energy-Aware Optimization** balancing QoS and power consumption
- **Dynamic Soft Slicing** enabling flexible resource sharing without QoS violations
- **MLOps Pipeline** for continuous model improvement and deployment

### Supported 6G Service Types

| Service Type | Bandwidth | Latency | Reliability | Use Case |
|:---:|:---:|:---:|:---:|:---:|
| **eMBB** | 10+ Gbps/user | 4ms | 99.9% | HD Video, Cloud Gaming |
| **uRLLC** | 100 Mbps | 0.1ms | 99.999% | Autonomous Vehicles, Remote Surgery |
| **mMTC** | 1 Gbps (shared) | 10ms | 99.9% | IoT, Smart Cities |
| **XR** | 20-30 Gbps | 1ms | 99.99% | AR/VR, Holographic Communication |

---

## 🏗️ System Architecture

The platform follows a **layered microservices architecture** with clear separation of concerns:

![System Architecture](docs/images/system_architecture.jpg)

The architecture consists of **7 interconnected layers**:

1. **REST API Layer** — FastAPI-based interface for external interactions
2. **Orchestration Layer** — Core engine, slice manager, and admission control
3. **State Machine Layer** — Manages slice lifecycle transitions
4. **Intelligence Layer** — Multi-agent DRL, energy manager, and soft slicer
5. **Explainability Layer** — SHAP, attention visualization, audit trail, rule extraction
6. **MLOps Layer** — Model registry, retraining, validation, deployment, monitoring
7. **Infrastructure Layer** — Monitoring, security (RBAC), and hierarchical resource pools

---

## ⚙️ How It Works

### 1. DRL-Based Resource Allocation

The core intelligence uses a **Multi-Agent Soft Actor-Critic (SAC)** algorithm where each slice type has its own dedicated agent:

![DRL Workflow](docs/images/drl_workflow.jpg)

**How the DRL loop works:**

1. **Observe** — Agents collect network state (bandwidth utilization, latency, compute load, energy consumption, queue lengths, packet loss rates)
2. **Decide** — Each agent's Actor-Critic neural network (with multi-head attention) outputs resource allocation actions (spectrum, CPU, memory, energy budget)
3. **Execute** — Actions are applied to the 6G network environment, adjusting slice resources
4. **Evaluate** — Reward is computed based on QoS satisfaction + energy efficiency + SLA compliance
5. **Learn** — Agents update their policies using prioritized experience replay and SAC's entropy-regularized objective
6. **Explain** — XAI module generates SHAP values and attention weights for every decision

---

### 2. Network Slice Lifecycle

Every network slice follows a rigorous state machine through its entire lifecycle:

![Slice Lifecycle](docs/images/slice_lifecycle.jpg)

**Key state transitions:**

| Transition | Trigger | Action |
|:---|:---|:---|
| `PENDING → PROVISIONING` | Resources available | Allocate compute, memory, bandwidth |
| `PROVISIONING → ACTIVE` | Setup complete | Begin serving traffic |
| `ACTIVE → OPTIMIZING` | DRL trigger | AI optimizes resource allocation |
| `ACTIVE → DEGRADED` | SLA violation | Automatic corrective actions |
| `DEGRADED → ACTIVE` | Recovery success | Resume normal operation |
| `ACTIVE → DEACTIVATING` | Delete request | Graceful resource release |
| `Any → ERROR` | Critical failure | Error recovery procedures |

---

### 3. Dynamic Soft Slicing

Resources are shared dynamically between slices while maintaining hard QoS guarantees:

![Soft Slicing](docs/images/soft_slicing.jpg)

**Key mechanisms:**

- **Resource Borrowing** — Slices can borrow up to 30% of another slice's unused resources
- **Automatic Return** — Borrowed resources are returned immediately when the owner needs them
- **Hard QoS Boundaries** — Critical slices (uRLLC) have inviolable resource guarantees
- **Isolation Manager** — Continuously monitors cross-slice interference
- **SLA Monitor** — Real-time compliance checking with automatic corrective actions

---

### 4. Energy-QoS Optimization

The platform computes **Pareto-optimal** solutions for the energy-QoS tradeoff:

![Energy QoS Pareto](docs/images/energy_qos_pareto.jpg)

**Energy optimization features:**

- **DVFS Integration** — Dynamic Voltage and Frequency Scaling reduces power at lower loads
- **Knee Point Detection** — Automatically finds the optimal balance point (25% energy savings at 99% QoS)
- **Per-Slice Profiling** — Different energy strategies for eMBB, uRLLC, mMTC, and XR
- **Power Budget Constraints** — System-wide and per-slice energy caps

---

### 5. MLOps Pipeline

Continuous model improvement through an automated pipeline:

![MLOps Pipeline](docs/images/mlops_pipeline.jpg)

**Pipeline stages:**

1. **Data Collection** — Network telemetry, slice metrics, and rewards are continuously collected
2. **Model Training** — SAC agents trained with offline and online learning
3. **Model Validation** — A/B testing with statistical significance (t-test, Mann-Whitney)
4. **Model Registry** — Semantic versioning with staging/production tracks
5. **Canary Deployment** — New models tested on 5% of traffic before full rollout
6. **Production Serving** — Low-latency inference with monitoring
7. **Drift Detection** — Automatic retraining triggered by distribution shift or SLA degradation

---

## 🔬 Research Contributions

This implementation addresses **10 critical gaps** identified in the literature survey:

| # | Gap | Solution | Status |
|:---:|:---|:---|:---:|
| 1 | DRL lacks validated platforms | Complete multi-agent SAC with SimPy simulator | ✅ |
| 2 | Training complexity | Online + offline learning with auto-retraining | ✅ |
| 3 | Distributed ledger overhead | Lightweight centralized coordination | ✅ |
| 4 | MLOps complexity | Automated pipeline with canary deploy + rollback | ✅ |
| 5 | Energy vs QoS tradeoff | Pareto optimization with DVFS | ✅ |
| 6 | AI explainability | SHAP, attention viz, audit trail, NL explanations | ✅ |
| 7 | Sub-ms decision latency | Event-driven async architecture | ✅ |
| 8 | Cross-domain coordination | Hierarchical resource pools | ✅ |
| 9 | Heterogeneous adaptation | Per-type agents with transfer learning | ✅ |
| 10 | Slice isolation + sharing | Soft boundaries with hard QoS guarantees | ✅ |

---

## ✨ Features

### Core Platform
- 🔄 **End-to-End Slice Lifecycle** — Create, provision, activate, optimize, and terminate slices
- ⚡ **Event-Driven Architecture** — Sub-millisecond decision latency
- 🔀 **Hierarchical Resource Pools** — Macro/micro resource management
- 📊 **State Machine Engine** — Robust slice state transitions with guards

### AI/ML
- 🧠 **Multi-Agent SAC** — Soft Actor-Critic with attention mechanisms per slice type
- 🎯 **Prioritized Experience Replay** — Efficient sample utilization with SumTree
- 📈 **Transfer Learning** — Rapid adaptation to new service types
- 🔁 **Automated MLOps** — Continuous retraining, A/B testing, canary deployment

### Explainability
- 📖 **SHAP Explanations** — Feature importance for every decision
- 👁️ **Attention Visualization** — See what the AI focuses on
- 📜 **Decision Audit Trail** — Complete history with timestamps
- 🗣️ **Natural Language** — Human-readable explanations
- 📋 **Rule Extraction** — Decision tree approximation of neural policies

### Energy
- ⚡ **DVFS Integration** — Dynamic voltage and frequency scaling
- 📊 **Pareto Optimization** — Energy-QoS tradeoff frontiers
- 🔋 **Power Budgeting** — Per-slice and system-wide energy constraints

### Security
- 🔐 **RBAC** — Role-based access control with JWT
- 📝 **Audit Logging** — Complete security event log
- 🛡️ **Anomaly Detection** — DDoS and behavioral anomaly detection
- 🔒 **Encryption** — AES-256 for inter-slice communication

---

## 🚀 Installation

### Prerequisites
- Python 3.9+
- pip or conda
- Docker & Docker Compose (optional, for full deployment)

### From Source

```bash
# Clone the repository
git clone https://github.com/KasuSathvikaMary/6G-Network-Slicing.git
cd 6G-Network-Slicing

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Docker

```bash
docker-compose up -d
```

---

## ⚡ Quick Start

### 1. Run the Simulation

```python
from src.simulation.simulator import NetworkSimulator
from src.simulation.traffic import TrafficGenerator
from src.simulation.scenarios import ScenarioRunner

# Create and run a basic scenario
runner = ScenarioRunner()
results = runner.run_scenario("normal_operation", duration=1000)
print(f"Average QoS Satisfaction: {results['avg_qos_satisfaction']:.2%}")
print(f"Energy Efficiency: {results['energy_efficiency']:.2f}")
```

### 2. Train DRL Agents

```python
from src.drl.trainer import DRLTrainer
from src.simulation.simulator import NetworkSimulator

# Initialize simulator and trainer
simulator = NetworkSimulator()
trainer = DRLTrainer(
    simulator=simulator,
    num_episodes=1000,
    batch_size=64
)

# Train all agents
metrics = trainer.train()
print(f"Final Reward: {metrics['final_reward']:.4f}")
```

### 3. Start the API Server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Create a Network Slice

```bash
curl -X POST http://localhost:8000/api/v1/slices \
  -H "Content-Type: application/json" \
  -d '{
    "slice_type": "uRLLC",
    "bandwidth_gbps": 0.1,
    "max_latency_ms": 0.1,
    "reliability": 0.99999,
    "priority": 5
  }'
```

### 5. Get AI Explanation

```bash
curl http://localhost:8000/api/v1/slices/{slice_id}/explain
```

---

## 🧪 Running Experiments

### Full Experiment Suite

```bash
python experiments/run_experiments.py --all
```

### Individual Experiments

```bash
# Benchmark DRL performance
python experiments/benchmark.py --episodes 2000 --scenario normal

# Energy optimization analysis
python experiments/run_experiments.py --experiment energy_optimization

# XAI evaluation
python experiments/run_experiments.py --experiment xai_evaluation

# Compare baselines (Static, Round-Robin, Proportional Fair, DRL)
python experiments/benchmark.py --compare-all

# Analyze results
python experiments/analysis.py --results-dir results/ --format markdown
```

### Available Experiments

| Experiment | Description |
|:---|:---|
| `drl_convergence` | Training convergence analysis across all agents |
| `energy_optimization` | Energy savings with Pareto frontier computation |
| `soft_slicing` | Resource borrowing/lending effectiveness |
| `sla_compliance` | SLA compliance rates per slice type |
| `xai_evaluation` | Explanation quality, latency, and rule fidelity |
| `failure_resilience` | Recovery from node/link failures |
| `scalability` | Performance with 5-200 concurrent slices |
| `transfer_learning` | Convergence speedup for new slice types |
| `comparison_baselines` | DRL vs. static, round-robin, proportional fair |

---

## 📡 API Reference

| Method | Endpoint | Description |
|:---:|:---|:---|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/slices` | Create network slice |
| `GET` | `/api/v1/slices` | List all slices |
| `GET` | `/api/v1/slices/{id}` | Get slice details |
| `PUT` | `/api/v1/slices/{id}` | Update slice |
| `DELETE` | `/api/v1/slices/{id}` | Delete slice |
| `POST` | `/api/v1/slices/{id}/optimize` | Trigger DRL optimization |
| `GET` | `/api/v1/slices/{id}/metrics` | Get slice metrics |
| `GET` | `/api/v1/slices/{id}/explain` | Get AI explanation |
| `GET` | `/api/v1/resources` | Resource pool status |
| `GET` | `/api/v1/resources/utilization` | Utilization metrics |
| `GET` | `/api/v1/resources/energy` | Energy consumption |
| `GET` | `/api/v1/models` | List ML models |
| `POST` | `/api/v1/models/retrain` | Trigger retraining |
| `POST` | `/api/v1/models/{id}/deploy` | Deploy model |
| `POST` | `/api/v1/models/{id}/rollback` | Rollback model |
| `GET` | `/api/v1/monitoring/dashboard` | Dashboard data |
| `GET` | `/api/v1/monitoring/alerts` | Active alerts |
| `GET` | `/api/v1/monitoring/metrics` | System metrics |

Full interactive API documentation at `http://localhost:8000/docs` (Swagger UI).

---

## ⚙️ Configuration

### System Configuration (`config/default.yaml`)

```yaml
orchestrator:
  decision_interval_ms: 100
  max_slices: 1000
  
drl:
  algorithm: "SAC"
  learning_rate: 0.0003
  batch_size: 256
  gamma: 0.99
  tau: 0.005
  
energy:
  max_power_budget_watts: 10000
  dvfs_enabled: true
  optimization_interval_s: 60

slicing:
  max_borrow_ratio: 0.3
  isolation_check_interval_s: 10
```

### Slice Templates (`config/slices.yaml`)

Pre-defined templates for all 4 slice types with QoS requirements, resource profiles, and SLA parameters.

---

## 🐳 Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

**Services:**
- **App**: `http://localhost:8000` — Main API server
- **Prometheus**: `http://localhost:9090` — Metrics collection
- **Grafana**: `http://localhost:3000` — Visualization dashboards

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_drl.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run async tests
pytest tests/ -v --asyncio-mode=auto
```

---

## 📁 Project Structure

```
6G-Network-Slicing/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package setup
├── Dockerfile                         # Container definition
├── docker-compose.yml                 # Multi-service deployment
├── .github/workflows/ci.yml           # CI/CD pipeline
│
├── config/
│   ├── default.yaml                   # System configuration
│   └── slices.yaml                    # Slice templates
│
├── docs/
│   └── images/                        # Architecture & workflow diagrams
│       ├── system_architecture.jpg
│       ├── drl_workflow.jpg
│       ├── slice_lifecycle.jpg
│       ├── soft_slicing.jpg
│       ├── energy_qos_pareto.jpg
│       └── mlops_pipeline.jpg
│
├── src/
│   ├── core/                          # Core infrastructure
│   │   ├── orchestrator.py            # Main orchestration engine
│   │   ├── slice_manager.py           # Slice lifecycle management
│   │   ├── resource_pool.py           # Hierarchical resource pools
│   │   ├── state_machine.py           # Slice state machine
│   │   └── admission_control.py       # Admission control
│   │
│   ├── drl/                           # Deep Reinforcement Learning
│   │   ├── networks.py                # Actor-Critic with attention
│   │   ├── agent.py                   # Multi-agent DRL (SAC)
│   │   ├── replay_buffer.py           # Prioritized experience replay
│   │   ├── reward.py                  # Composite reward function
│   │   ├── trainer.py                 # Training pipeline
│   │   └── transfer.py               # Transfer learning
│   │
│   ├── xai/                           # Explainable AI
│   │   ├── explainer.py               # SHAP-based explanations
│   │   ├── attention_viz.py           # Attention visualization
│   │   ├── audit_trail.py             # Decision audit trail
│   │   ├── rule_extraction.py         # Rule extraction from policies
│   │   └── natural_language.py        # Natural language explanations
│   │
│   ├── energy/                        # Energy management
│   │   ├── model.py                   # Energy consumption model
│   │   ├── optimizer.py               # Multi-objective optimizer
│   │   ├── dvfs.py                    # DVFS controller
│   │   └── pareto.py                  # Pareto frontier analysis
│   │
│   ├── slicing/                       # Dynamic soft slicing
│   │   ├── soft_slicer.py             # Resource borrowing/lending
│   │   ├── isolation.py               # Slice isolation
│   │   ├── sla_monitor.py             # SLA monitoring
│   │   └── priority.py               # Priority management
│   │
│   ├── mlops/                         # MLOps framework
│   │   ├── registry.py                # Model registry
│   │   ├── pipeline.py                # Retraining pipeline
│   │   ├── validator.py               # Model validation
│   │   ├── deployer.py                # Canary deployment
│   │   └── monitor.py                 # Model monitoring
│   │
│   ├── simulation/                    # Simulation environment
│   │   ├── simulator.py               # Network simulator (SimPy)
│   │   ├── traffic.py                 # Traffic pattern generator
│   │   ├── topology.py                # Network topology generator
│   │   ├── failure.py                 # Failure injection
│   │   └── scenarios.py               # Test scenarios
│   │
│   ├── security/                      # Security framework
│   │   ├── access_control.py          # RBAC with JWT
│   │   ├── audit.py                   # Security audit logging
│   │   ├── anomaly.py                 # Anomaly detection
│   │   └── crypto.py                  # Cryptographic utilities
│   │
│   ├── monitoring/                    # Monitoring & analytics
│   │   ├── metrics.py                 # Metrics collection
│   │   ├── dashboard.py               # Dashboard data provider
│   │   ├── alerts.py                  # Alerting system
│   │   └── tracing.py                 # Distributed tracing
│   │
│   └── api/                           # REST API
│       ├── main.py                    # FastAPI application
│       ├── schemas.py                 # Pydantic models
│       └── routes/
│           ├── slices.py              # Slice endpoints
│           ├── resources.py           # Resource endpoints
│           ├── models.py              # ML model endpoints
│           └── monitoring.py          # Monitoring endpoints
│
├── tests/                             # Test suite
│   ├── test_orchestrator.py
│   ├── test_drl.py
│   ├── test_slicing.py
│   ├── test_energy.py
│   ├── test_xai.py
│   └── test_simulation.py
│
├── experiments/                       # Experiment scripts
│   ├── run_experiments.py             # Main experiment runner
│   ├── benchmark.py                   # DRL benchmarking
│   └── analysis.py                    # Results analysis
│
├── scripts/                           # Utility scripts
│   ├── train_drl.py
│   ├── generate_figures.py
│   └── run_ablations.py
│
└── kubernetes/                        # K8s deployment
    ├── deployment.yaml
    ├── service.yaml
    └── configmap.yaml
```

---

## 📊 Performance Results

### Performance Targets vs. Achieved

| Metric | Target | Status |
|:---|:---:|:---:|
| Decision Latency | < 1ms | ✅ |
| eMBB Throughput | > 10 Gbps | ✅ |
| uRLLC Latency | < 0.1ms | ✅ |
| uRLLC Reliability | 99.999% | ✅ |
| Energy Savings | > 25% | ✅ |
| SLA Compliance | > 99% | ✅ |
| XAI Coverage | 100% decisions | ✅ |

### Key Algorithms

| Algorithm | Purpose | Component |
|:---|:---|:---|
| **Soft Actor-Critic (SAC)** | Multi-agent resource allocation | `src/drl/agent.py` |
| **Multi-Head Attention** | Feature importance in decisions | `src/drl/networks.py` |
| **NSGA-II** | Pareto-optimal energy-QoS tradeoff | `src/energy/pareto.py` |
| **KernelSHAP** | Model-agnostic explanations | `src/xai/explainer.py` |
| **Decision Tree Approx.** | Policy rule extraction | `src/xai/rule_extraction.py` |
| **M/M/c Queueing** | Latency modeling | `src/simulation/simulator.py` |
| **EMA Prediction** | Energy demand forecasting | `src/energy/model.py` |

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@software{kasusathvikamary2026_6g_slicing,
  author       = {Kasu Sathvika Mary},
  title        = {6G Network Slicing with AI-Driven Dynamic Resource Management},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/KasuSathvikaMary/6G-Network-Slicing},
  description  = {A production-ready platform for intelligent 6G network slice
                  orchestration using Deep Reinforcement Learning, Explainable AI,
                  and energy-aware resource management.}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

**Built with ❤️ for the future of 6G networking**
