## ABSTRACT

This paper presents a hierarchical multi-agent reinforcement learning (MARL) framework for dynamic energy management in large-scale power systems with high penetration of distributed energy resources (DER). The proposed system operates on a 117-bus tri-area SuperGrid constructed by merging three IEEE 39-bus (New England) test systems into a mesh-interconnected topology with eight inter-area tie-lines, 30 synchronous generators, and an integrated DER fleet comprising 205 MW of solar photovoltaic capacity, 250 MW of wind generation, 120 MWh of battery energy storage, 200 electric vehicle chargers, and 36.5 MW of demand response. A three-level Centralized Training with Decentralized Execution (CTDE) architecture deploys 13 Proximal Policy Optimization (PPO) agents organized hierarchically: a single central coordination agent governing inter-area power balance and tie-line flow targets, three microgrid-level agents managing area-specific generation dispatch and DER curtailment, and nine task-specific sub-agents controlling individual inverter setpoints, renewable curtailment fractions, and flexible load participation. Each agent operates within a dedicated Gymnasium environment with level-appropriate observation and action spaces, and receives a bounded, multi-objective reward signal composed of weighted components for frequency regulation, voltage profile quality, economic efficiency, DER utilization, and constraint violation penalties. The simulation backbone integrates Newton-Raphson AC power flow with a six-strategy convergence fallback cascade, IEEE 421.5-2016 compliant excitation and power system stabilizer models, TGOV1 governor-turbine dynamics, PI-based automatic generation control with Indian Electricity Grid Code (IEGC) deadband, multi-stage under-frequency load shedding protection, and quadratic economic dispatch with locational marginal pricing. Simulation fidelity is validated through 216 automated tests confirming power flow convergence in four Newton-Raphson iterations, 100% bus voltage compliance within the IEEE 0.95--1.05 pu band, active power losses of 0.709% of total generation, and zero thermal violations across all 117 buses and 113 transmission lines. Small-signal stability analysis on the full 330-state system confirms correct eigenvalue decomposition and IEEE-compliant stability margins. The hierarchical agent architecture, multi-objective reward formulation, constraint validation framework, and eight IEEE-aligned evaluation scenarios with four baseline controllers are presented as an integrated, reproducible platform for investigating MARL-based grid management under realistic operating conditions.

---

## I. INTRODUCTION

### A. Motivation and Problem Context

The accelerating integration of distributed energy resources into electric power systems represents one of the defining operational challenges of the contemporary energy transition. Global installed solar photovoltaic capacity exceeded 1.6 TW by the end of 2023 [1], while wind power capacity surpassed 1 TW [2], collectively displacing synchronous generation and fundamentally altering the dynamic characteristics of interconnected grids. In India, the Central Electricity Authority has targeted 500 GW of non-fossil fuel generation capacity by 2030 [3], a trajectory that will subject the Indian grid to unprecedented levels of inverter-based resource penetration, reduced rotational inertia, and bidirectional power flows across distribution and transmission boundaries.

These developments impose simultaneous and often conflicting requirements on grid operators. Frequency regulation becomes more demanding as the displacement of synchronous machines reduces the aggregate system inertia available to arrest rate-of-change-of-frequency (RoCoF) excursions following contingency events [4]. Voltage management grows more complex as spatially distributed, weather-dependent generation introduces rapid fluctuations in reactive power flows [5]. Economic dispatch must accommodate near-zero marginal cost renewable generation while maintaining sufficient conventional reserves for ramping and balancing services [6]. Concurrently, the proliferation of controllable flexible loads—electric vehicle charging, battery energy storage systems, and demand response programs—expands the action space available to grid operators but also increases the combinatorial complexity of real-time optimization [7].

Traditional approaches to these challenges rely on hierarchical, rule-based control architectures: primary droop response at the generator level, secondary automatic generation control (AGC) at the area level, and tertiary economic dispatch at the system level [8]. While these methods are well-understood and mathematically tractable for conventional generation portfolios, they exhibit fundamental limitations in the presence of high DER penetration. Specifically, they assume quasi-static operating conditions, rely on linearised system models that degrade in accuracy under large disturbances, and lack the capacity to learn non-linear relationships between control actions and system responses across stochastic operating conditions [9].

### B. Reinforcement Learning for Power System Control

Reinforcement learning (RL) has emerged as a promising paradigm for power system control, offering the ability to learn optimal policies directly from interaction with complex, non-linear environments without requiring explicit system models [10]. Single-agent deep RL approaches have demonstrated success in voltage regulation [11], economic dispatch [12], and emergency load shedding [13], typically operating on simplified grid models with limited state and action dimensionality.

However, the application of single-agent RL to realistic multi-area power systems encounters the curse of dimensionality: a 117-bus system with 30 generators and diverse DER yields a state space exceeding 300 continuous dimensions and an action space of 60--100 continuous controls. This scale renders single-agent learning intractable due to sample inefficiency and the difficulty of credit assignment across spatially distributed subsystems [14]. Multi-agent reinforcement learning (MARL) addresses this challenge by decomposing the global problem into cooperative sub-problems, where each agent operates on a local observation and action space while collectively optimizing a shared objective [15].

The Centralized Training with Decentralized Execution (CTDE) paradigm [16] is particularly well-suited to power systems, where centralised coordination signals (such as AGC area control error) naturally coexist with decentralised local controllers. Under CTDE, agents share information during training to develop coordinated strategies, but execute their policies independently using only local observations at inference time—mirroring the operational reality of geographically distributed grid controllers communicating through a supervisory energy management system.

Despite this conceptual alignment, the application of hierarchical MARL to physically detailed, standards-compliant power system simulations remains limited. The majority of existing MARL-based grid management studies operate on reduced network models (e.g., IEEE 14-bus or 30-bus systems) [17], employ simplified power flow approximations (e.g., DC power flow or linearised sensitivity factors) [18], and omit critical dynamic phenomena including excitation system transients, governor-turbine interactions, and protection relay operations [19]. Furthermore, few existing platforms provide integrated evaluation frameworks with standardised baselines, IEEE-compliant stability metrics, and contingency scenarios that would enable reproducible comparison across methods [20].

### C. Proposed Approach

This paper presents a comprehensive framework that addresses the gaps identified above through four integrated subsystems:

**1) A Physically Detailed Simulation Environment.** The grid model is a 117-bus tri-area SuperGrid constructed from three merged instances of the IEEE 39-bus New England test system [21], interconnected via eight tie-lines in a mesh topology that ensures N-1 security. The simulation integrates full AC power flow via Newton-Raphson iteration with a six-strategy convergence fallback cascade, per-generator electromechanical dynamics (swing equation with symplectic Euler integration, IEEE IEEET1 excitation system per IEEE Std 421.5-2016 [22], TGOV1 governor-turbine model, and PSS1A power system stabilizer), PI-based automatic generation control with Indian Electricity Grid Code (IEGC) [23] deadband of 0.03 Hz, multi-stage under-frequency load shedding protection (49.5/49.2/49.0 Hz), and a five-type DER fleet (solar PV, wind, battery storage, EV charging, and demand response) compliant with IEEE 1547-2018 [24] interconnection requirements including low-voltage and high-voltage ride-through. Stochastic operating conditions are generated through Ornstein-Uhlenbeck processes for wind speed and cloud cover, and diurnal profiles for load variation.

**2) A Hierarchical Multi-Agent Architecture.** Thirteen PPO agents [25] are organised in a three-level hierarchy that mirrors the operational structure of interconnected power systems. A single central coordination agent observes global state summaries and inter-area tie-line flows (24-dimensional observation) and outputs area generation biases and tie-line flow targets (6-dimensional action). Three microgrid-level agents each observe their respective area's power balance, voltage profile, DER status, and dynamic state (24-dimensional observation per area) and output generator dispatch fractions and DER curtailment commands (8-dimensional action). Nine task-specific sub-agents—three each for inverter control, renewable energy curtailment, and flexible load management—operate on narrow 10-dimensional observations and produce fine-grained setpoints for individual assets. The central and microgrid agents employ two-layer [256, 256] multilayer perceptron policy networks, while sub-agents use [128, 128] architectures, all trained via PPO with Stable-Baselines3 [26].

**3) A Multi-Objective, Constraint-Aware Reward Formulation.** Each hierarchy level receives a distinct reward signal composed of weighted sub-components, all individually bounded to [−1, 1]. The central agent's reward prioritises frequency regulation (35%), tie-line flow balance (25%), and action smoothness (20%), with secondary terms for global stability (12%) and economic efficiency (8%). Microgrid-level rewards emphasise voltage profile quality (30%), economic generation cost via quadratic cost curves (25%), constraint violation penalties (20%), DER utilisation (15%), and action smoothness (10%). Sub-agent rewards are tailored to their functional roles: voltage support and power tracking for inverter agents; generation maximisation with curtailment minimisation for renewable agents; and demand response effectiveness with customer comfort for load agents. Constraint satisfaction is enforced through explicit penalty terms linked to a `GridConstraintValidator` module that monitors IEGC frequency limits (49.5--50.5 Hz), IEEE voltage limits (0.95--1.05 pu per bus), line thermal loading, and generator ramp rates continuously during training.

**4) A Reproducible Evaluation Framework.** Eight evaluation scenarios aligned with IEEE and NERC standards have been implemented, spanning normal operation, high renewable penetration, N-1 generator and tie-line contingencies, sudden load ramping, area islanding, price spikes, and low-inertia conditions. Four baseline controllers—no-control (fixed setpoints), droop (R = 5%), merit-order economic dispatch, and PI-based AGC—are provided for comparative benchmarking. An automated metrics collection pipeline captures per-timestep data across frequency stability, voltage quality, economic performance, reliability, and environmental indices, and an IEEE publication-quality plotting module generates 18 figure types formatted for two-column journal layout.

### D. Contributions

The principal contributions of this work are as follows:

1. **An open-source, 117-bus tri-area simulation environment** that integrates AC power flow, IEEE-standard dynamic models, five DER types with IEEE 1547-2018 compliance, quadratic economic dispatch, and protection relay logic into a unified Gymnasium-compatible RL interface. Unlike existing RL-for-grid platforms that operate on simplified or linearised models [27], [28], this environment preserves the non-linear coupling between control actions and physical system response at a fidelity level suitable for standards-compliant stability assessment.

2. **A three-level hierarchical CTDE multi-agent architecture** with 13 PPO agents whose observation spaces, action spaces, and reward functions are systematically mapped to the operational hierarchy of interconnected power systems. To the best of the authors' knowledge, this represents the first publicly available MARL framework that combines hierarchical agent decomposition with physically detailed, multi-area grid simulation at the 100+ bus scale [29], [30].

3. **A multi-objective reward design methodology** in which all sub-reward components are bounded to [−1, 1], individually interpretable, and composed via configurable weight vectors, enabling transparent trade-off analysis between frequency regulation, voltage quality, economic efficiency, DER utilization, and constraint satisfaction across hierarchy levels.

4. **An IEEE-aligned evaluation protocol** comprising eight contingency and stress scenarios (IEEE 1547-2018, NERC TPL-001-4, IEGC), four classical baseline controllers, approximately 40 per-timestep metrics aligned with IEEE 1366 reliability and NERC BAL-001 frequency response standards, and automated generation of publication-quality figures.

5. **A validated simulation platform** verified through 216 automated tests confirming topological correctness (117 buses, 30 generators, 8 tie-lines), power flow convergence (4 Newton-Raphson iterations, 0.709% active losses), 100% voltage compliance within the IEEE 0.95--1.05 pu band across all buses, IEEE 421.5-2016 stability margin compliance, and small-signal eigenvalue analysis on a 330-state system.

### E. Paper Organization

The remainder of this paper is organized as follows. Section II reviews related work in reinforcement learning for power systems, multi-agent approaches, and existing simulation platforms. Section III describes the simulation environment, including the SuperGrid topology, power flow engine, dynamic models, DER integration, and economic dispatch formulation. Section IV presents the hierarchical multi-agent methodology, detailing the CTDE training procedure, observation and action space design, and reward function formulation. Section V reports simulation validation results and preliminary training outcomes. Section VI discusses limitations and directions for future work. Section VII concludes the paper.

---

## REFERENCES

[1] International Renewable Energy Agency (IRENA), "Renewable Capacity Statistics 2024," Abu Dhabi, 2024. [Online]. Available: https://www.irena.org/Publications/2024/Mar/Renewable-capacity-statistics-2024

[2] Global Wind Energy Council (GWEC), "Global Wind Report 2024," Brussels, Belgium, 2024. [Online]. Available: https://gwec.net/global-wind-report-2024/

[3] Central Electricity Authority (CEA), Government of India, "National Electricity Plan (Volume I) — Generation," New Delhi, 2023. [Online]. Available: https://cea.nic.in/national-electricity-plan/

[4] ENTSO-E, "High Penetration of Power Electronic Interfaced Power Sources and the Potential Contribution of Grid Forming Converters," Technical Report, Brussels, 2020.

[5] A. T. Procopiou and L. F. Ochoa, "Voltage control in PV-rich LV networks without remote monitoring," *IEEE Trans. Power Syst.*, vol. 32, no. 2, pp. 1224–1236, Mar. 2017.

[6] F. Ueckerdt, R. Brecha, and G. Luderer, "Analyzing major challenges of wind and solar variability in power systems," *Renew. Energy*, vol. 81, pp. 1–10, Sep. 2015.

[7] P. Siano, "Demand response and smart grids — A survey," *Renew. Sustain. Energy Rev.*, vol. 30, pp. 461–478, Feb. 2014.

[8] P. Kundur, *Power System Stability and Control*. New York, NY, USA: McGraw-Hill, 1994.

[9] E. Ela, M. Milligan, A. Bloom, A. Botterud, A. Townsend, and T. Levin, "Evolution of wholesale electricity market design with increasing levels of renewable generation," National Renewable Energy Laboratory (NREL), Tech. Rep. NREL/TP-5D00-61765, 2014.

[10] Z. Zhang, D. Zhang, and R. C. Qiu, "Deep reinforcement learning for power system applications: An overview," *CSEE J. Power Energy Syst.*, vol. 6, no. 1, pp. 213–225, Mar. 2020.

[11] Q. Yang, G. Wang, A. Sadeghi, G. B. Giannakis, and J. Sun, "Two-timescale voltage control in distribution grids using deep reinforcement learning," *IEEE Trans. Smart Grid*, vol. 11, no. 3, pp. 2313–2323, May 2020.

[12] Y. Zhou, W. S. Lee, R. Rong, and D. Ranaweera, "Deep reinforcement learning for economic dispatch: A review and case study," *Appl. Energy*, vol. 348, p. 121512, Oct. 2023.

[13] H. Huang, Y. Zhang, and M. Zhou, "Adaptive frequency control of microgrid based on deep reinforcement learning," in *Proc. IEEE Power Energy Soc. Gen. Meeting (PESGM)*, Montreal, QC, Canada, 2020, pp. 1–5.

[14] W. Wang, N. Yu, Y. Gao, and J. Shi, "Safe off-policy deep reinforcement learning algorithm for volt-VAR control in power distribution systems," *IEEE Trans. Smart Grid*, vol. 11, no. 4, pp. 3008–3018, Jul. 2020.

[15] K. Zhang, Z. Yang, and T. Basar, "Multi-agent reinforcement learning: A selective overview of theories and algorithms," in *Handbook of Reinforcement Learning and Control*, K. G. Vamvoudakis, Y. Wan, F. L. Lewis, and D. Cansever, Eds. Cham, Switzerland: Springer, 2021, pp. 321–384.

[16] R. Lowe, Y. Wu, A. Tamar, J. Harb, P. Abbeel, and I. Mordatch, "Multi-agent actor-critic for mixed cooperative-competitive environments," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, Long Beach, CA, USA, 2017, pp. 6379–6390.

[17] S. Wang, J. Duan, D. Shi, C. Xu, H. Li, R. Diao, and Z. Wang, "A data-driven multi-agent autonomous voltage control framework using deep reinforcement learning," *IEEE Trans. Power Syst.*, vol. 35, no. 6, pp. 4644–4654, Nov. 2020.

[18] B. Stott, J. Jardim, and O. Alsac, "DC power flow revisited," *IEEE Trans. Power Syst.*, vol. 24, no. 3, pp. 1290–1300, Aug. 2009.

[19] F. Milano, *Power System Modelling and Scripting*. Berlin, Germany: Springer, 2010.

[20] A. Marot, B. Donnot, G. Dulac-Arnold *et al.*, "Learning to run a power network challenge for training topology controllers," *Electric Power Syst. Res.*, vol. 189, p. 106635, Dec. 2020.

[21] T. Athay, R. Podmore, and S. Virmani, "A practical method for the direct analysis of transient stability," *IEEE Trans. Power App. Syst.*, vol. PAS-98, no. 2, pp. 573–584, Mar./Apr. 1979.

[22] IEEE Power and Energy Society, "IEEE Recommended Practice for Excitation System Models for Power System Stability Studies," *IEEE Std 421.5-2016 (Revision of IEEE Std 421.5-2005)*, 2016.

[23] Central Electricity Regulatory Commission (CERC), "Indian Electricity Grid Code (IEGC)," Regulation No. 8 of 2023, New Delhi, India, 2023. [Online]. Available: https://cercind.gov.in/

[24] IEEE Standards Association, "IEEE Standard for Interconnection and Interoperability of Distributed Energy Resources with Associated Electric Power Systems Interfaces," *IEEE Std 1547-2018 (Revision of IEEE Std 1547-2003)*, 2018.

[25] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," *arXiv preprint arXiv:1707.06347*, 2017.

[26] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, "Stable-Baselines3: Reliable reinforcement learning implementations," *J. Mach. Learn. Res.*, vol. 22, no. 268, pp. 1–8, 2021.

[27] A. Marot, B. Donnot, C. Romero *et al.*, "Learning to run a power network challenge: A retrospective," in *Proc. NeurIPS Competition Track*, 2021, pp. 112–132.

[28] J. Vazquez-Canteli and Z. Nagy, "Reinforcement learning for demand response: A review of algorithms and modeling techniques," *Appl. Energy*, vol. 235, pp. 1072–1089, Feb. 2019.

[29] J. R. Vazquez-Canteli, J. Kämpf, G. Henze, and Z. Nagy, "CityLearn v1.0: An OpenAI Gym environment for demand response with deep reinforcement learning," in *Proc. ACM Int. Conf. Future Energy Syst. (e-Energy)*, Phoenix, AZ, USA, 2019, pp. 356–357.

[30] Z. Yan and Y. Xu, "A multi-agent deep reinforcement learning method for cooperative load frequency control of a multi-area power system," *IEEE Trans. Power Syst.*, vol. 35, no. 6, pp. 4599–4608, Nov. 2020.
