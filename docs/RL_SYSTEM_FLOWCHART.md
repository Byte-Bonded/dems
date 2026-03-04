# DEMS — RL Implementation & System Simulation Flowchart

## Full Architecture

```mermaid
flowchart TB
    %% ── Entry ──
    TRAIN["🚀 train.py<br/>(Entry Point)"]
    CONFIG["⚙️ TrainingConfig<br/>steps · lr · batch · device"]

    TRAIN --> CONFIG
    TRAIN --> HT

    %% ── Hierarchical Trainer ──
    subgraph TRAINER["Hierarchical Trainer  (CTDE)"]
        HT["HierarchicalTrainer<br/>.train() loop"]
        HT -->|reset / step| COORD
    end

    %% ── Coordinator ──
    subgraph COORDINATOR["Multi-Agent Step Coordinator"]
        COORD["MultiAgentStepCoordinator<br/>orchestrates hierarchy each timestep"]
        COORD -->|"1 · global obs (24-D)"| CENTRAL
        COORD -->|"2 · area obs (24-D ×3)"| MG
        COORD -->|"3 · narrow obs (10-D ×9)"| SUBS
        COORD -->|"4 · combined actions"| PHYSICS
        COORD -->|"5 · collect rewards"| HT
    end

    %% ── Central Agent ──
    subgraph CENTRAL_BOX["Central Coordination Layer"]
        CENTRAL["CentralCoordEnv<br/>1 × PPO Agent  ‹256,256›"]
        COBS["Obs 24-D<br/>3 area summaries · 6 tie-lines<br/>sys freq · gen · load · losses · time"]
        CACT["Action 6-D<br/>area_A/B/C bias · tie_AB/BC/AC targets<br/>(±200 MW)"]
        CREWARD["CentralReward<br/>freq 35% · tie-line 25%<br/>stability 12% · economy 8%<br/>smoothness 20%"]
        CENTRAL --- COBS
        CENTRAL --- CACT
        CENTRAL --- CREWARD
    end

    %% ── Microgrid Agents ──
    subgraph MG_BOX["Microgrid Agent Layer  (×3 areas: A · B · C)"]
        MG["MicrogridEnv<br/>3 × PPO Agent  ‹256,256›"]
        MOBS["Obs 24-D per area<br/>power · voltages · DER<br/>dynamics · environment · time"]
        MACT["Action (variable-D)<br/>gen setpoints · solar/wind curtail<br/>EV util · DR curtail"]
        MREWARD["MicrogridReward<br/>voltage 30% · economy 25%<br/>DER util 15% · constraints 20%<br/>smoothness 10%"]
        MG --- MOBS
        MG --- MACT
        MG --- MREWARD
    end

    %% ── Sub-Agents ──
    subgraph SUB_BOX["Sub-Agent Layer  (×9 total — 3 per area)"]
        SUBS["Sub-Envs<br/>9 × PPO Agent  ‹128,128›"]

        INV["InverterSubEnv ×3<br/>per-gen fractional setpoints"]
        REN["RenewableSubEnv ×3<br/>solar/wind curtailment fractions"]
        LOAD["LoadSubEnv ×3<br/>EV utilisation + DR curtailment"]

        SUBS --> INV
        SUBS --> REN
        SUBS --> LOAD
    end

    %% ── Physics Engine ──
    subgraph PHYSICS_BOX["Physics Engine  (Shared Simulation)"]
        PHYSICS["PhysicsEngine.step()"]

        PROFILES["StochasticProfileGenerator<br/>random solar · wind · load per episode"]
        GRID["SuperGrid  (117-Bus)<br/>3 × IEEE 39-bus merged<br/>30 generators · 138+ lines<br/>8 tie-lines (mesh)"]
        PF["PowerFlowRunner<br/>Newton-Raphson AC<br/>FDBX / Q-relaxed fallback"]
        DYN["DynamicsCoordinator<br/>N substeps × 20 ms = 5 s per step"]
        DER["DERManager"]
        ECON["EconomicsEngine<br/>quadratic cost curves · TOU pricing<br/>carbon rates · merit-order"]
        PROT["ProtectionRelay<br/>UFLS 3-stage · OFGT · Voltage trip"]

        PHYSICS --> PROFILES
        PHYSICS --> GRID
        PHYSICS --> PF
        PHYSICS --> DYN
        PHYSICS --> DER
        PHYSICS --> ECON
        PHYSICS --> PROT
    end

    %% ── DER Fleet ──
    subgraph DER_BOX["DER Fleet  (per area)"]
        SOLAR["☀️ Solar PV<br/>~205 MW<br/>irradiance-based"]
        WIND["💨 Wind Turbines<br/>~250 MW<br/>IEC 61400 cubic curve"]
        EV["🔌 EV Charging<br/>~15 MW  (200 chargers)<br/>smart charging"]
        DR["📉 Demand Response<br/>~36.5 MW curtailable<br/>incentive-based"]

        DER --> SOLAR
        DER --> WIND
        DER --> EV
        DER --> DR
    end

    %% ── Dynamics Models ──
    subgraph DYN_BOX["Electromechanical Dynamics  (30 generators)"]
        SWING["Swing Equation<br/>symplectic Euler"]
        AVR["Excitation / AVR<br/>IEEE IEEET1 + OEL"]
        GOV["Governor-Turbine<br/>TGOV1 + valve limits"]
        PSS["PSS1A<br/>washout + lead-lag"]
        AGC["AGC<br/>ACE-based PI + deadband"]
        DLOAD["Dynamic Load Model<br/>ZIP freq/voltage dependent"]

        DYN --> SWING
        DYN --> AVR
        DYN --> GOV
        DYN --> PSS
        DYN --> AGC
        DYN --> DLOAD
    end

    %% ── Constraints ──
    CONSTRAINTS["GridConstraintValidator<br/>IEGC freq 49.5–50.5 Hz<br/>IEEE volt 0.95–1.05 pu<br/>thermal limits · gen ramps"]
    PHYSICS --> CONSTRAINTS
    CONSTRAINTS -->|ConstraintReport| MREWARD
    CONSTRAINTS -->|ConstraintReport| CREWARD

    %% ── Evaluation Branch ──
    subgraph EVAL_BOX["Evaluation Pipeline"]
        EVAL["run_eval.py<br/>N episodes × K scenarios"]
        BASELINES["Baselines<br/>No-Control · Droop (R=5%)<br/>Merit-Order · PI-AGC"]
        SCENARIOS["Scenarios<br/>Base · High Renewable · N-1 Gen<br/>N-1 Tie · Load Ramp · Island<br/>Price Spike · Low Inertia"]
        METRICS["MetricsCollector<br/>~40 fields per timestep"]
        PLOTS["IEEEPlotter<br/>Bode · Nyquist · Eigenvalue<br/>time-series · LaTeX tables"]

        EVAL --> BASELINES
        EVAL --> SCENARIOS
        EVAL --> METRICS
        METRICS --> PLOTS
    end

    HT -->|"trained model"| EVAL

    %% ── Monitoring ──
    subgraph MON_BOX["Monitoring Stack"]
        PROM["Prometheus<br/>port 9090"]
        GRAF["Grafana Dashboards<br/>Real-Time Sim · RL Optimized"]
        SIM_GEN["demo_metrics.py<br/>port 9091  (dems_*)"]
        RL_GEN["dummy_metrics_generator.py<br/>port 9092  (rl_*)"]

        SIM_GEN --> PROM
        RL_GEN --> PROM
        PROM --> GRAF
    end

    PHYSICS -.->|"live metrics"| SIM_GEN
    EVAL -.->|"RL metrics"| RL_GEN

    %% ── Styling ──
    classDef entry fill:#2196F3,color:#fff,stroke:#1565C0
    classDef trainer fill:#9C27B0,color:#fff,stroke:#6A1B9A
    classDef central fill:#FF9800,color:#fff,stroke:#E65100
    classDef mg fill:#4CAF50,color:#fff,stroke:#2E7D32
    classDef sub fill:#00BCD4,color:#fff,stroke:#00838F
    classDef physics fill:#F44336,color:#fff,stroke:#C62828
    classDef der fill:#8BC34A,color:#000,stroke:#558B2F
    classDef dyn fill:#FF5722,color:#fff,stroke:#BF360C
    classDef eval fill:#607D8B,color:#fff,stroke:#37474F
    classDef mon fill:#795548,color:#fff,stroke:#4E342E
    classDef constraint fill:#E91E63,color:#fff,stroke:#880E4F

    class TRAIN,CONFIG entry
    class HT trainer
    class CENTRAL,COBS,CACT,CREWARD central
    class MG,MOBS,MACT,MREWARD mg
    class SUBS,INV,REN,LOAD sub
    class PHYSICS,PROFILES,GRID,PF physics
    class DER,SOLAR,WIND,EV,DR der
    class DYN,SWING,AVR,GOV,PSS,AGC,DLOAD dyn
    class ECON,PROT physics
    class EVAL,BASELINES,SCENARIOS,METRICS,PLOTS eval
    class PROM,GRAF,SIM_GEN,RL_GEN mon
    class CONSTRAINTS constraint
```

---

## Training Step Sequence

```mermaid
sequenceDiagram
    participant T as train.py
    participant HT as HierarchicalTrainer
    participant C as Coordinator
    participant CA as Central Agent
    participant MA as Microgrid Agents ×3
    participant SA as Sub-Agents ×9
    participant PE as PhysicsEngine

    T->>HT: train(total_steps, eval_freq)
    loop Each Episode
        HT->>C: reset()
        C->>PE: reset() → new stochastic profiles
        C-->>HT: initial observations

        loop Each Timestep (288 steps = 24h)
            HT->>CA: predict(global_obs_24D)
            CA-->>HT: action_6D (area biases + tie targets)

            HT->>MA: predict(area_obs_24D) ×3
            MA-->>HT: actions (gen setpoints + DER) ×3

            HT->>SA: predict(narrow_obs_10D) ×9
            SA-->>HT: fine actions ×9

            HT->>C: step(all_actions)
            C->>PE: apply actions → AC power flow → dynamics
            PE-->>C: new state + constraint reports
            C-->>HT: observations + rewards (13 agents)

            HT->>HT: store transitions in rollout buffers
        end

        HT->>HT: PPO update (all 13 agents)
    end

    Note over HT: Periodic eval with deterministic actions
    HT-->>T: trained models saved to checkpoints/
```

---

## Physics Engine Per-Step Pipeline

```mermaid
flowchart LR
    A["Apply Stochastic<br/>Load Profiles"] --> B["Adjust Reactive<br/>Resources"]
    B --> C["AC Power Flow<br/>(Newton-Raphson)"]
    C -->|converged| D["Step Dynamics<br/>(N × 20 ms)"]
    C -->|failed| C2["Fallback: FDBX /<br/>Q-relaxed"]
    C2 --> D
    D --> E["Protection<br/>Relay Check"]
    E --> F["Restore Base<br/>Loads"]
    F --> G["Termination<br/>Check"]

    subgraph DYN["Dynamics Sub-steps"]
        D1["Swing Equation"] --> D2["AVR (IEEET1)"]
        D2 --> D3["Governor (TGOV1)"]
        D3 --> D4["PSS (PSS1A)"]
        D4 --> D5["AGC"]
        D5 --> D6["Dynamic Loads"]
    end

    D --> DYN
```

---

## Observation & Action Summary

```mermaid
flowchart LR
    subgraph OBS["Observation Spaces"]
        O1["Central: 24-D<br/>area summaries · tie-lines<br/>freq · gen · load · time"]
        O2["Microgrid: 24-D<br/>area power · voltages · DER<br/>dynamics · environment · time"]
        O3["Sub-agent: 10-D<br/>local v/f · unit output<br/>area context · time"]
    end

    subgraph ACT["Action Spaces"]
        A1["Central: 6-D<br/>3 area biases<br/>3 tie-line targets"]
        A2["Microgrid: variable-D<br/>gen setpoints<br/>DER curtailment"]
        A3["Sub-agent: per-unit<br/>inverter / renewable /<br/>load fractions"]
    end

    subgraph REW["Reward Components"]
        R1["Central<br/>freq 35% · tie 25%<br/>stability 12% · econ 8%<br/>smooth 20%"]
        R2["Microgrid<br/>voltage 30% · econ 25%<br/>DER 15% · constraint 20%<br/>smooth 10%"]
        R3["Sub-agent<br/>voltage support<br/>power tracking<br/>comfort"]
    end

    O1 --> A1 --> R1
    O2 --> A2 --> R2
    O3 --> A3 --> R3
```

---

## Grid Topology

```mermaid
flowchart TB
    subgraph AREA_A["Area A — IEEE 39-bus"]
        GA["10 Generators<br/>nuclear · coal · gas · hydro"]
        BA["39 Buses"]
        DA["DER: Solar + Wind + EV + DR"]
    end

    subgraph AREA_B["Area B — IEEE 39-bus"]
        GB["10 Generators<br/>nuclear · coal · gas · hydro"]
        BB["39 Buses"]
        DB["DER: Solar + Wind + EV + DR"]
    end

    subgraph AREA_C["Area C — IEEE 39-bus"]
        GC["10 Generators<br/>nuclear · coal · gas · hydro"]
        BC["39 Buses"]
        DC["DER: Solar + Wind + EV + DR"]
    end

    AREA_A <-->|"3 tie-lines"| AREA_B
    AREA_B <-->|"3 tie-lines"| AREA_C
    AREA_A <-->|"2 tie-lines"| AREA_C
```

---

## Evaluation Pipeline

```mermaid
flowchart LR
    MODEL["Trained RL Model"] --> EVAL["run_eval.py"]
    BASE["Baselines<br/>No-Control · Droop<br/>Merit-Order · PI-AGC"] --> EVAL

    EVAL --> SC["Scenarios"]

    subgraph SCEN["8 Test Scenarios"]
        S1["Base Case (24h)"]
        S2["High Renewable (80%+)"]
        S3["N-1 Generator"]
        S4["N-1 Tie-line"]
        S5["Load Ramp (+20%)"]
        S6["Islanding"]
        S7["Price Spike"]
        S8["Low Inertia"]
    end

    SC --> SCEN

    EVAL --> MC["MetricsCollector<br/>~40 fields / step"]
    MC --> IEEE["IEEEPlotter"]

    subgraph OUT["Outputs"]
        P1["Time-series Plots"]
        P2["Bode / Nyquist"]
        P3["Eigenvalue Analysis"]
        P4["LaTeX Comparison Tables"]
    end

    IEEE --> OUT
```
