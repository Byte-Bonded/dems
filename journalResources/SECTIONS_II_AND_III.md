## II. SYSTEM MODEL AND PROBLEM FORMULATION

This section presents the mathematical models underlying the 117-bus tri-area SuperGrid simulation environment, including the network topology, AC power flow formulation, electromechanical dynamics, distributed energy resource models, economic dispatch, and the formal multi-objective optimization problem.

### A. Network Topology

The test system is a 117-bus tri-area SuperGrid constructed by merging three instances of the IEEE 39-bus (New England) test system [21]. Each area $k \in \mathcal{K} = \{A, B, C\}$ consists of 39 buses indexed with an offset $o_k$:

$$
o_A = 0, \quad o_B = 39, \quad o_C = 78
$$

such that bus $j$ in area $k$ maps to global index $o_k + j$ for $j \in \{0, 1, \ldots, 38\}$. The merged network comprises:

- $N_b = 117$ buses;
- $N_g = 30$ synchronous generators (10 per area, of which 9 are dispatchable and 1 models the external grid slack reference);
- $N_l = 113$ transmission lines and $N_t = 33$ transformers;
- $N_d = 63$ loads (21 per area);
- $N_{\text{tie}} = 8$ inter-area tie-lines.

The tie-line topology forms a mesh interconnection:

$$
\mathcal{T} = \{T_{AB}^{(1)}, T_{AB}^{(2)}, T_{AB}^{(3)}, T_{BC}^{(1)}, T_{BC}^{(2)}, T_{BC}^{(3)}, T_{AC}^{(1)}, T_{AC}^{(2)}\}
$$

with three parallel paths between areas $A$ and $B$, three between $B$ and $C$, and two direct paths between $A$ and $C$. Each tie-line is parameterized by series impedance $(r + jx)$ per km, length $l$ (km), charging capacitance $c$ (nF/km), and thermal rating $S_{\max}$ (MVA). The three-way mesh ensures N-1 security: the loss of any single tie-line does not disconnect any area.

### B. AC Power Flow

The system operating point is determined by solving the AC power flow equations via Newton-Raphson iteration. For each bus $i \in \{1, \ldots, N_b\}$, the complex power injection is:

$$
S_i = P_i + jQ_i = V_i \sum_{j=1}^{N_b} Y_{ij}^* V_j^*
$$

where $V_i = |V_i| e^{j\theta_i}$ is the complex bus voltage and $Y_{ij}$ is the $(i,j)$-th element of the bus admittance matrix $\mathbf{Y}_{\text{bus}} \in \mathbb{C}^{N_b \times N_b}$. Expanding into real and imaginary components yields the mismatch equations:

$$
\Delta P_i = P_{G,i} - P_{D,i} - |V_i| \sum_{j=1}^{N_b} |V_j| \left( G_{ij} \cos\theta_{ij} + B_{ij} \sin\theta_{ij} \right) = 0
$$

$$
\Delta Q_i = Q_{G,i} - Q_{D,i} - |V_i| \sum_{j=1}^{N_b} |V_j| \left( G_{ij} \sin\theta_{ij} - B_{ij} \cos\theta_{ij} \right) = 0
$$

where $\theta_{ij} = \theta_i - \theta_j$, $G_{ij} + jB_{ij} = Y_{ij}$, and $P_{G,i}$, $Q_{G,i}$, $P_{D,i}$, $Q_{D,i}$ are the active and reactive generation and demand at bus $i$, respectively.

The Newton-Raphson method iteratively solves the linearized system:

$$
\begin{bmatrix} \Delta \boldsymbol{\theta} \\ \Delta |\mathbf{V}| \end{bmatrix}^{(\nu+1)} =
-\mathbf{J}^{-1} \begin{bmatrix} \Delta \mathbf{P} \\ \Delta \mathbf{Q} \end{bmatrix}^{(\nu)}
$$

where $\mathbf{J}$ is the Jacobian matrix composed of partial derivatives $\partial P / \partial \theta$, $\partial P / \partial |V|$, $\partial Q / \partial \theta$, and $\partial Q / \partial |V|$. Convergence is declared when $\max(|\Delta P_i|, |\Delta Q_i|) < \epsilon$ with tolerance $\epsilon = 10^{-8}$ MVA.

To ensure robust convergence under the wide range of operating conditions generated during RL exploration, the solver employs a six-strategy automatic fallback cascade:

1. Primary Newton-Raphson with configured tolerances;
2. DC-initialised Newton-Raphson;
3. Reactive power-limit-relaxed Newton-Raphson;
4. Flat-start with relaxed Q-limits;
5. Relaxed convergence tolerance ($10\epsilon$);
6. DC power flow as a linear last resort.

Active power losses on each branch $(i, j)$ are computed as:

$$
P_{\text{loss}}^{(ij)} = R_{ij} |I_{ij}|^2
$$

and the total system loss is $P_{\text{loss}} = \sum_{(i,j)} P_{\text{loss}}^{(ij)}$.

### C. Electromechanical Dynamics

The dynamic behaviour of each synchronous generator is modelled using the classical swing equation augmented by field flux dynamics, IEEE-standard excitation, governor-turbine, and power system stabilizer models. The simulation advances all generators over $N_{\text{sub}}$ sub-steps of duration $\Delta t_{\text{sub}} = 20$ ms within each environment step of $\Delta T = 5$ s, corresponding to $N_{\text{sub}} = 250$ sub-steps.

#### 1) Swing Equation

For generator $i$ with inertia constant $H_i$ (s) and damping coefficient $D_i$ (pu), the rotor dynamics are:

$$
\frac{d\omega_i}{dt} = \frac{1}{2H_i} \left( P_{m,i} - P_{e,i} - D_i (\omega_i - 1) \right)
$$

$$
\frac{d\delta_i}{dt} = \omega_0 (\omega_i - 1)
$$

where $\omega_i$ is the rotor speed in per-unit, $\delta_i$ is the rotor angle (rad), $P_{m,i}$ and $P_{e,i}$ are the mechanical and electrical power in per-unit on the machine MVA base, and $\omega_0 = 2\pi f_0 = 100\pi$ rad/s for a 50 Hz system. Integration employs symplectic Euler: the speed $\omega_i$ is updated first, and the new $\omega_i$ is then used to update $\delta_i$, which preserves the Hamiltonian structure and provides unconditional numerical stability.

The system frequency is computed as the centre-of-inertia (COI) frequency:

$$
f_{\text{sys}} = f_0 \cdot \frac{\sum_{i=1}^{N_g} H_i \omega_i}{\sum_{i=1}^{N_g} H_i}
$$

#### 2) Field Flux Dynamics

The transient internal EMF $E'_{q,i}$ evolves according to:

$$
T'_{d0,i} \frac{dE'_{q,i}}{dt} = E_{fd,i} - E'_{q,i}
$$

where $T'_{d0,i}$ is the d-axis transient open-circuit time constant and $E_{fd,i}$ is the field voltage output from the excitation system. This equation couples the exciter output into the generator electromechanical model.

#### 3) Excitation System (IEEE IEEET1)

The excitation system follows the IEEE Std 421.5-2016 Section 5.1 model (IEEET1), comprising a voltage transducer with time constant $T_R$, a voltage regulator with gain $K_A$ and time constant $T_A$, a stabilising feedback path with gain $K_F$ and time constant $T_F$, an exciter with gain $K_E$ and time constant $T_E$, and a saturation function:

$$
S_E(E_{fd}) = A_{\text{SE}} \exp(B_{\text{SE}} |E_{fd}|)
$$

The voltage error signal is:

$$
V_e = V_{\text{ref}} - V_{t,\text{filt}} + V_{\text{PSS}} - V_{\text{fb}}
$$

where $V_{\text{ref}}$ is the voltage reference setpoint, $V_{t,\text{filt}}$ is the filtered terminal voltage, $V_{\text{PSS}}$ is the PSS supplementary signal, and $V_{\text{fb}}$ is the stabilising feedback signal computed as:

$$
V_{\text{fb}} = \frac{K_F}{T_F} E_{fd} - V_f, \qquad T_F \frac{dV_f}{dt} = \frac{K_F}{T_F} E_{fd} - V_f
$$

The regulator output is:

$$
T_A \frac{dV_R}{dt} = K_A V_e - V_R, \qquad V_R^{\min} \leq V_R \leq V_R^{\max}
$$

The exciter equation with saturation is:

$$
\frac{T_E}{K_E + S_E} \frac{dE_{fd}}{dt} = V_R - (K_E + S_E) E_{fd}
$$

An Over-Excitation Limiter (OEL) per IEEE 421.5-2016 Section 6 progressively reduces $V_{\text{ref}}$ when $|E_{fd}|$ exceeds the thermal limit $E_{fd}^{\max}$ for a duration exceeding the OEL delay $\tau_{\text{OEL}}$.

**Default parameters:** $K_A = 200$, $T_A = 0.02$ s, $T_R = 0.02$ s, $K_E = 1.0$, $T_E = 0.5$ s, $K_F = 0.03$, $T_F = 1.0$ s, $V_R^{\max} = 5.0$ pu, $V_R^{\min} = -5.0$ pu, $A_{\text{SE}} = 0.0039$, $B_{\text{SE}} = 1.555$.

#### 4) Governor-Turbine (TGOV1)

The governor responds to speed deviations via droop characteristic $R$ and a valve positioning time constant $T_G$:

$$
T_G \frac{dP_g}{dt} = P_{\text{ref}} - \frac{\Delta\omega}{R} - P_g
$$

Valve position changes are rate-limited:

$$
\dot{P}_g^{\min} \leq \frac{dP_g}{dt} \leq \dot{P}_g^{\max}
$$

with default rates $\dot{P}_g^{\max} = 0.1$ pu/s and $\dot{P}_g^{\min} = -0.1$ pu/s. The turbine output follows with time constant $T_T$:

$$
T_T \frac{dP_m}{dt} = P_g - P_m
$$

The mechanical power delivered to the shaft includes turbine damping:

$$
P_{m,\text{out}} = P_m - D_t \Delta\omega
$$

where $D_t = 0.05$ pu is the turbine damping coefficient.

**Default parameters:** $R = 0.05$, $T_G = 0.2$ s, $T_T = 0.5$ s, $P^{\max} = 1.5$ pu, $P^{\min} = 0.0$ pu, $D_t = 0.05$ pu.

#### 5) Power System Stabilizer (PSS1A)

The PSS follows the IEEE Std 421.5-2016 Section 8.1 (PSS1A) model with speed deviation $\Delta\omega$ as input. The transfer function comprises a washout filter and two lead-lag stages:

$$
G_{\text{PSS}}(s) = K_{\text{PSS}} \cdot \frac{sT_w}{1 + sT_w} \cdot \frac{1 + sT_{\text{lead},1}}{1 + sT_{\text{lag},1}} \cdot \frac{1 + sT_{\text{lead},2}}{1 + sT_{\text{lag},2}}
$$

The washout filter rejects steady-state (DC) components, while each lead-lag stage has unity DC gain. The PSS output is clamped to $[V_{\text{PSS}}^{\min}, V_{\text{PSS}}^{\max}] = [-0.2, 0.2]$ pu.

**Default parameters:** $K_{\text{PSS}} = 5.0$, $T_w = 1.41$ s, $T_{\text{lead},1} = T_{\text{lead},2} = 0.154$ s, $T_{\text{lag},1} = T_{\text{lag},2} = 0.033$ s.

#### 6) Automatic Generation Control

AGC operates as a PI controller on the Area Control Error (ACE):

$$
\text{ACE}_k = \Delta P_{\text{tie},k} + \beta_k \Delta f
$$

where $\Delta P_{\text{tie},k}$ is the net tie-line flow error for area $k$, $\beta_k = 1000$ MW/Hz is the frequency bias coefficient, and $\Delta f = f_{\text{sys}} - f_0$. A deadband of $\pm 0.03$ Hz (IEGC standard) suppresses AGC response to normal frequency fluctuations. The AGC output is:

$$
u_{\text{AGC}}(t) = -K_{\text{AGC}} \cdot \text{ACE}(t) - \frac{1}{T_{\text{AGC}}} \int_0^t \text{ACE}(\tau)\, d\tau
$$

with $K_{\text{AGC}} = 0.5$ and $T_{\text{AGC}} = 4.0$ s. The AGC output is clamped to $\pm 400$ MW and distributed to participating generators via normalized participation factors $\alpha_i$ ($\sum_i \alpha_i = 1$), with per-generator setpoint changes limited to 0.05 pu per AGC interval.

#### 7) Numerical Integration

All first-order dynamic blocks (transducer, regulator, exciter, governor, turbine, PSS stages) are integrated using the implicit trapezoidal method:

$$
x^{(n+1)} = \frac{x^{(n)} (2T - \Delta t) + 2u^{(n+1)} \Delta t}{2T + \Delta t}
$$

for the canonical form $T \dot{x} = u - x$. This method is unconditionally A-stable for any $\Delta t > 0$ and $T > 0$, ensuring numerical robustness during transient events.

### D. Dynamic Load Model

Loads are modelled using the ZIP (constant impedance, constant current, constant power) formulation with frequency dependence:

$$
P_d(V, f) = P_0 \left( Z_P V^2 + I_P V + P_P \right) \left( 1 + K_{pf} \frac{\Delta f}{f_0} \right)
$$

$$
Q_d(V, f) = Q_0 \left( Z_Q V^2 + I_Q V + P_Q \right) \left( 1 + K_{qf} \frac{\Delta f}{f_0} \right)
$$

where $V$ is the bus voltage magnitude in per-unit, $\Delta f = f - f_0$, and the default coefficients are $Z_P = 0.4$, $I_P = 0.3$, $P_P = 0.3$, $Z_Q = 0.5$, $I_Q = 0.3$, $P_Q = 0.2$, $K_{pf} = 1.5$, and $K_{qf} = -1.0$.

### E. Protection System

The protection relay subsystem implements the Indian Grid Code (IEGC) requirements:

- **Under-Frequency Load Shedding (UFLS):** Three stages at 49.5 Hz (10% load shed), 49.2 Hz (15%), and 49.0 Hz (20%), each with a trip delay of 0.2 s.
- **Over-Frequency Generation Trip (OFGT):** Generator tripping at 50.5 Hz sustained for 0.5 s.
- **Voltage Protection:** Under-voltage trip at 0.90 pu and over-voltage trip at 1.10 pu, each with a 2.0 s delay. Warning alarms activate at 0.95 pu and 1.05 pu respectively.

### F. Distributed Energy Resources

Each area hosts four types of DER, all compliant with IEEE 1547-2018 interconnection requirements including Category III low-voltage ride-through (LVRT), high-voltage ride-through (HVRT), anti-islanding detection (2 s), and active power ramp rate limiting.

#### 1) Solar Photovoltaic

Output is determined by solar irradiance $G$ (W/m$^2$):

$$
P_{\text{solar}} = \eta \cdot A_{\text{panel}} \cdot G / 10^6 \quad \text{(MW)}
$$

with installed capacity of approximately 205 MW across three areas. Output is subject to curtailment fractions $\kappa_{\text{solar}} \in [0, 1]$ as determined by the renewable sub-agent.

#### 2) Wind Generation

Wind turbine output follows the IEC 61400 cubic power curve:

$$
P_{\text{wind}} = \begin{cases}
0, & v < v_{\text{ci}} \\
P_{\text{rated}} \cdot \dfrac{v^3 - v_{\text{ci}}^3}{v_r^3 - v_{\text{ci}}^3}, & v_{\text{ci}} \leq v \leq v_r \\
P_{\text{rated}}, & v_r < v \leq v_{\text{co}} \\
0, & v > v_{\text{co}}
\end{cases}
$$

where $v$ is wind speed (m/s), $v_{\text{ci}}$, $v_r$, and $v_{\text{co}}$ are the cut-in, rated, and cut-out wind speeds, and $P_{\text{rated}} \approx 250$ MW aggregate across three areas. Subject to curtailment fraction $\kappa_{\text{wind}} \in [0, 1]$.

#### 3) Electric Vehicle Charging

The EV charging fleet comprises 200 chargers with smart charging capability. Aggregate load is modelled as:

$$
P_{\text{EV}} = N_{\text{chargers}} \cdot P_{\text{charger}} \cdot u_{\text{EV}}
$$

where $u_{\text{EV}} \in [0, 1]$ is the EV utilisation fraction controlled by the load sub-agent.

#### 4) Demand Response

Curtailable load programs offer up to 36.5 MW of flexible capacity:

$$
P_{\text{DR}} = P_{\text{baseline}} \cdot (1 - \kappa_{\text{DR}})
$$

where $\kappa_{\text{DR}} \in [0, 1]$ is the demand curtailment fraction, $P_{\text{baseline}}$ is the baseline load, and the curtailable fraction is determined by contract terms. Incentive pricing is market-based.

### G. Economic Model

Generator operating costs follow quadratic cost curves:

$$
C_i(P_i) = a_i + b_i P_i + c_i P_i^2 \quad (\$/\text{h})
$$

where $a_i$ ($/h) is the fixed cost, $b_i$ ($/MWh) is the linear coefficient, and $c_i$ ($/MWh$^2$) is the quadratic coefficient. The marginal cost at output $P_i$ is:

$$
\text{MC}_i(P_i) = b_i + 2c_i P_i \quad (\$/\text{MWh})
$$

Cost curve parameters are sourced from the MATPOWER case39 dataset [27], with fuel type assignments: nuclear and coal for base-load units (generators 0--3 per area), gas combined-cycle for mid-merit units (generators 4--6), gas peakers (generators 7--8), and hydro for the slack bus equivalent (generator 9).

Carbon emissions are computed using EPA eGRID 2022 factors: coal 0.95 tCO$_2$/MWh, gas 0.41 tCO$_2$/MWh, oil 0.73 tCO$_2$/MWh, and zero for hydro and nuclear. Time-of-Use (TOU) pricing follows a CERC India-derived tariff structure with higher tariffs during peak hours, and Real-Time Pricing (RTP) responds to instantaneous supply-demand balance.

### H. Problem Formulation

The dynamic energy management problem is formulated as a cooperative multi-agent Markov decision process (MA-MDP). Let $\mathcal{N} = \{1, 2, \ldots, 13\}$ denote the set of agents. At each discrete timestep $t \in \{0, 1, \ldots, T-1\}$ with $T = 288$ steps (representing 24 hours at 5-minute intervals), each agent $n$ observes a local observation $o_t^n \in \mathcal{O}^n$, selects an action $a_t^n \in \mathcal{A}^n$ according to its policy $\pi^n(a_t^n | o_t^n)$, and receives a scalar reward $r_t^n$. The joint action $\mathbf{a}_t = (a_t^1, \ldots, a_t^{13})$ is applied to the shared physics engine, which transitions the global state $s_t \in \mathcal{S}$ to $s_{t+1}$ according to the deterministic dynamics described in Sections II-B through II-F.

The objective is to find a joint policy $\boldsymbol{\pi}^* = (\pi^{1*}, \ldots, \pi^{13*})$ that maximises the expected discounted cumulative return for all agents:

$$
\boldsymbol{\pi}^* = \arg\max_{\boldsymbol{\pi}} \sum_{n=1}^{13} \mathbb{E}_{\boldsymbol{\pi}} \left[ \sum_{t=0}^{T-1} \gamma^t r_t^n \right]
$$

subject to the physical and regulatory constraints:

$$
f^{\min} \leq f_{\text{sys}}(t) \leq f^{\max}, \quad \forall\, t \qquad \text{(IEGC: 49.5--50.5 Hz)}
$$

$$
V_i^{\min} \leq |V_i(t)| \leq V_i^{\max}, \quad \forall\, i, t \qquad \text{(IEEE: 0.95--1.05 pu)}
$$

$$
|S_{ij}(t)| \leq S_{ij}^{\max}, \quad \forall\, (i,j) \in \mathcal{L}, t \qquad \text{(thermal limits)}
$$

$$
|P_{g,i}(t) - P_{g,i}(t-1)| \leq \Delta P_i^{\max}, \quad \forall\, i \in \mathcal{G}, t \qquad \text{(ramp rates)}
$$

$$
P_{g,i}^{\min} \leq P_{g,i}(t) \leq P_{g,i}^{\max}, \quad \forall\, i \in \mathcal{G}, t \qquad \text{(gen limits)}
$$

where $\gamma = 0.99$ is the discount factor, $\mathcal{L}$ is the set of branches, $\mathcal{G}$ is the set of generators, and the constraint limits follow IEGC and IEEE standards as specified in Section II-E.

---

## III. PROPOSED METHODOLOGY

This section details the hierarchical multi-agent reinforcement learning architecture, including the agent organisation, observation and action space design, reward function formulations, the CTDE training procedure, and the constraint handling mechanism.

### A. Hierarchical Agent Architecture

The 13 PPO agents are organised in a three-level hierarchy that mirrors the operational structure of multi-area power systems:

**Level 1 — Central Coordination (1 agent).** The central agent operates at the system level, analogous to a national load dispatch centre. It observes aggregated inter-area state and produces coordination signals—area generation biases and tie-line flow targets—that guide the lower-level agents without directly commanding individual generators.

**Level 2 — Microgrid Management (3 agents).** One microgrid agent per area ($k \in \{A, B, C\}$) manages area-level generation dispatch and DER curtailment. Each microgrid agent receives coordination context from the central agent and has direct control over generator active power setpoints and DER utilisation fractions within its area.

**Level 3 — Sub-Agents (9 agents).** Three specialised sub-agents per area refine specific control tasks:

- *Inverter sub-agent*: controls per-generator fractional power setpoints;
- *Renewable sub-agent*: determines solar and wind curtailment fractions;
- *Load sub-agent*: manages EV utilisation and demand response curtailment.

This decomposition reduces the per-agent action space dimensionality, facilitates credit assignment, and enables each agent to specialise in a coherent subset of the control problem.

### B. Observation Spaces

Each hierarchy level observes a feature vector constructed by a dedicated observation builder. All observation values are normalised to approximately unit scale and clipped to $[-2, 2]$ to prevent gradient explosion. NaN values arising from non-converged power flow are replaced with zero.

#### 1) Central Agent Observation ($\mathcal{O}^{\text{central}} \subset \mathbb{R}^{24}$)

The central agent observation aggregates inter-area state:

| Index | Feature | Normalisation |
|-------|---------|---------------|
| 0--3 | Area A summary: $P_G/5000$, $P_D/5000$, $P_{\text{int}}/500$, $\Delta f$ | per-area |
| 4--7 | Area B summary (same layout) | per-area |
| 8--11 | Area C summary (same layout) | per-area |
| 12--17 | Tie-line flows (6 lines): $P_{\text{tie}}/500$ | per-line |
| 18 | System frequency: $f_{\text{sys}}/50$ | — |
| 19 | Total generation: $P_G^{\text{tot}}/15000$ | — |
| 20 | Total load: $P_D^{\text{tot}}/15000$ | — |
| 21 | Total losses: $P_{\text{loss}}/1000$ | — |
| 22 | Hour of day: $h/24$ | — |
| 23 | Episode progress: $t/T$ | — |

#### 2) Microgrid Agent Observation ($\mathcal{O}^{\text{mg}} \subset \mathbb{R}^{24}$)

Each microgrid agent observes its own area's detailed state:

| Index | Feature | Normalisation |
|-------|---------|---------------|
| 0 | Area generation: $P_G^k / 5000$ | MW |
| 1 | Area load: $P_D^k / 5000$ | MW |
| 2 | Net interchange: $P_{\text{int}}^k / 500$ | MW |
| 3 | Generation headroom: $\sum(P_{\max} - P) / 1000$ | MW |
| 4 | Downward reserve: $\sum(P - P_{\min}) / 1000$ | MW |
| 5 | Area losses: $P_{\text{loss}}^k / 200$ | MW |
| 6 | Generators online: $n_g / 10$ | count |
| 7 | Average generator loading fraction | pu |
| 8 | Maximum line loading: $L_{\max} / 100$ | % |
| 9--11 | Voltage statistics: $\bar{V}$, $V_{\min}$, $V_{\max}$ | pu |
| 12 | Solar output: $P_{\text{solar}} / 100$ | MW |
| 13 | Wind output: $P_{\text{wind}} / 100$ | MW |
| 14 | EV load: $P_{\text{EV}} / 50$ | MW |
| 15 | DR curtailment fraction | pu |
| 16 | Frequency: $f / 50$ | Hz |
| 17 | Frequency deviation: $\text{clip}(\Delta f, \pm 1)$ | Hz |
| 18 | Average rotor speed | pu |
| 19--21 | Environment: irradiance$/1000$, wind speed$/25$, load scale | — |
| 22--23 | Time: hour$/24$, step$/T$ | — |

#### 3) Sub-Agent Observations ($\mathcal{O}^{\text{sub}} \subset \mathbb{R}^{10}$)

Each sub-agent type has a tailored 10-dimensional observation:

**Inverter sub-agent:**

$$
o^{\text{inv}} = [V_{\text{local}},\; \Delta f,\; P_g/500,\; Q_g/200,\; P_{\max}/500,\; P_g/P_{\max},\; V_{\min}^k,\; V_{\max}^k,\; h/24,\; t/T]
$$

**Renewable sub-agent:**

$$
o^{\text{ren}} = [G/G_{\max},\; P/P_{\text{rated}},\; P_{\text{rated}}/100,\; \kappa,\; V_{\text{local}},\; \Delta f,\; h/24,\; t/T,\; 0,\; 0]
$$

**Load sub-agent:**

$$
o^{\text{load}} = [P_D/5000,\; P_{\text{EV}}/50,\; \kappa_{\text{DR}},\; V_{\text{local}},\; \Delta f,\; \sigma_{\text{load}},\; h/24,\; t/T,\; 0,\; 0]
$$

where $G$ is solar irradiance or wind speed, $\kappa$ is the current curtailment fraction, $\sigma_{\text{load}}$ is the load scaling factor, and the trailing zeros are reserved for future features.

### C. Action Spaces

#### 1) Central Agent ($\mathcal{A}^{\text{central}} \subset [-1, 1]^6$)

$$
a^{\text{central}} = [\underbrace{b_A, b_B, b_C}_{\text{area biases}}, \underbrace{\tau_{AB}, \tau_{BC}, \tau_{AC}}_{\text{tie-line targets}}]
$$

The area bias signals $b_k \in [-1, 1]$ indicate the desired direction and magnitude of generation adjustment for each area. Tie-line target signals $\tau \in [-1, 1]$ are scaled to $\pm 200$ MW to set inter-area power transfer targets. These signals are not applied directly to generators; they serve as coordination context for the microgrid agents.

#### 2) Microgrid Agent ($\mathcal{A}^{\text{mg}} \subset [0, 1]^{d_k}$)

The action dimension $d_k$ varies by area and depends on the number of controllable elements:

$$
d_k = n_{\text{gen}}^k + n_{\text{solar}}^k + n_{\text{wind}}^k + n_{\text{EV}}^k + n_{\text{DR}}^k
$$

Each action component represents a fractional setpoint: generator power fractions $P_g / P_{\max} \in [0, 1]$, solar and wind curtailment fractions $\kappa \in [0, 1]$, EV utilisation fraction $u_{\text{EV}} \in [0, 1]$, and DR curtailment fraction $\kappa_{\text{DR}} \in [0, 1]$.

#### 3) Sub-Agent Actions ($\mathcal{A}^{\text{sub}} \subset [0, 1]^{d_{\text{sub}}}$)

Each sub-agent outputs fine-grained control signals with dimension $d_{\text{sub}} \leq 4$, corresponding to per-unit fractional setpoints within their domain:

- **Inverter:** per-generator fractional setpoints;
- **Renewable:** per-resource curtailment fractions;
- **Load:** EV utilisation and DR curtailment fractions.

### D. Reward Functions

All reward functions are designed to produce bounded scalar signals $r \in [-1, 1]$, ensuring stable gradient magnitudes across hierarchy levels. Each reward is a weighted sum of individually bounded sub-components.

#### 1) Central Agent Reward

$$
r^{\text{central}} = w_f \cdot r_f + w_{\text{tie}} \cdot r_{\text{tie}} + w_s \cdot r_s + w_e \cdot r_e + w_{\text{sm}} \cdot r_{\text{sm}}
$$

with default weights $w_f = 0.35$, $w_{\text{tie}} = 0.25$, $w_s \cdot 0.6 + w_s \cdot 0.4 = 0.20$ (split between stability and economy), and $w_{\text{sm}}$ absorbing the remainder.

**Frequency sub-reward:**

$$
r_f = 1 - \min\left( \left( \frac{|\Delta f|}{0.5} \right)^2, 1 \right)
$$

penalising frequency deviations quadratically within the IEGC band (49.5--50.5 Hz). If power flow did not converge, $r_f = -1$.

**Tie-line flow sub-reward:**

$$
r_{\text{tie}} = 1 - \min\left( \frac{\sum_{j} |P_{\text{tie},j}|}{1000}, 1 \right)
$$

penalising total absolute unscheduled interchange.

**Stability sub-reward:**

$$
r_s = \begin{cases} 1, & \text{if no protection trips} \\ \max(-1, 1 - 0.5 \cdot n_{\text{trips}}), & \text{otherwise} \end{cases}
$$

**Action smoothness sub-reward:**

$$
r_{\text{sm}} = 1 - \min\left( \frac{\|a_t - a_{t-1}\|_1 / d}{0.3}, 1 \right)
$$

where $d$ is the action dimension and $\|\cdot\|_1 / d$ is the mean absolute action change.

#### 2) Microgrid Agent Reward

$$
r^{\text{mg}} = 0.30 \cdot r_V + 0.25 \cdot r_C + 0.15 \cdot r_{\text{DER}} + 0.20 \cdot r_{\text{con}} + 0.10 \cdot r_{\text{sm}}
$$

**Voltage profile sub-reward:**

$$
r_V = \frac{1}{2} \left[ \max\left(0, 1 - \frac{|V_{\min} - 1|}{0.05}\right) + \max\left(0, 1 - \frac{|V_{\max} - 1|}{0.05}\right) \right]
$$

If $V_{\min} < 0.95$ pu:

$$
r_V^{\min} = -\frac{|V_{\min} - 0.95|}{0.10}
$$

and analogously for $V_{\max} > 1.05$ pu, yielding negative penalty.

**Economic cost sub-reward:**

When an economics engine is available:

$$
r_C = 1 - \min\left( \frac{C_{\text{total}}}{C_{\text{budget}}}, 2.0 \right)
$$

with $C_{\text{budget}} = 25{,}000$ \$/h. Otherwise, a loss-based proxy is used: $r_C = 1 - \min(P_{\text{loss}} / (0.03 \cdot P_G), 1)$.

**DER utilisation sub-reward:**

$$
r_{\text{DER}} = \min\left( \frac{P_{\text{solar}} + P_{\text{wind}}}{P_G^k}, 1 \right)
$$

**Constraint penalty sub-reward:**

$$
r_{\text{con}} = 1 - \min\left( \frac{\sigma_{\text{total}}}{5}, 1 \right)
$$

where $\sigma_{\text{total}} = \sum_{v} \sigma_v$ is the total normalized severity from the `GridConstraintValidator`, each violation $v$ having severity $\sigma_v \in [0, 1]$.

#### 3) Sub-Agent Rewards

**Inverter sub-agent:**

$$
r^{\text{inv}} = 0.5 \left( 1 - \min\left(\frac{|V_{\text{local}} - 1|}{0.05}, 1\right) \right) + 0.5 \left( 1 - \min\left(\frac{|P - P_{\text{target}}|}{|P_{\text{target}}|}, 1\right) \right)
$$

**Renewable sub-agent:**

$$
r^{\text{ren}} = 0.7 \cdot \min\left( \frac{P_{\text{output}}}{P_{\text{rated}}}, 1 \right) + 0.3 \cdot \mathbb{1}[V_{\text{local}} \in [0.95, 1.05]]
$$

**Load sub-agent:**

$$
r^{\text{load}} = 0.4 \cdot (1 - \kappa_{\text{DR}}) + 0.3 \cdot \mathbb{1}[V_{\text{local}} \in [0.95, 1.05]] + 0.3 \cdot r_{\text{freq}}
$$

where $r_{\text{freq}} = \min(2\kappa_{\text{DR}}, 1)$ during under-frequency events ($\Delta f < -0.2$ Hz), otherwise $r_{\text{freq}} = 1 - \kappa_{\text{DR}}$. This formulation encourages load curtailment during frequency emergencies while rewarding customer comfort under normal conditions.

### E. Centralised Training with Decentralised Execution

Training follows the Centralized Training with Decentralized Execution (CTDE) paradigm [16]. All 13 agents share a single `PhysicsEngine` instance through the `MultiAgentStepCoordinator`, which orchestrates each timestep in the following sequence:

1. The central agent observes the global state $o_t^{\text{central}}$ and produces coordination signals $a_t^{\text{central}}$.
2. Each microgrid agent observes its area-local state $o_t^{\text{mg},k}$ and produces area-level actions $a_t^{\text{mg},k}$.
3. Each sub-agent observes its narrow state $o_t^{\text{sub},m}$ and produces fine-grained actions $a_t^{\text{sub},m}$.
4. All actions are aggregated and applied to the physics engine, which executes one timestep (AC power flow + $N_{\text{sub}}$ dynamic sub-steps).
5. The new global state $s_{t+1}$ is computed, and each agent receives its level-specific observation and reward.

During training, agents are optimized using a round-robin schedule: each hierarchy level trains for a configurable number of steps (default 4,096) before the next level takes its turn. This approach prevents any single level from dominating gradient updates while maintaining consistent environment dynamics. The `StandaloneEnvWrapper` enables single-agent training by querying other agents for their (non-training) actions at each step, effectively treating co-agents as part of the environment.

At execution time, each agent uses only its local observation to select actions—no inter-agent communication is required, and the trained policies can be deployed to geographically distributed controllers.

### F. PPO Algorithm Configuration

All agents employ the Proximal Policy Optimization (PPO) algorithm [25] as implemented in Stable-Baselines3 [26], with the following hyperparameters:

**TABLE XI: PPO Hyperparameters by Hierarchy Level**

| Parameter | Central | Microgrid | Sub-Agent |
|-----------|---------|-----------|-----------|
| Policy network | MLP [256, 256] | MLP [256, 256] | MLP [128, 128] |
| Learning rate $\alpha$ | $3 \times 10^{-4}$ | $3 \times 10^{-4}$ | $3 \times 10^{-4}$ |
| Rollout steps $n_{\text{steps}}$ | 2,048 | 2,048 | 1,024 |
| Mini-batch size | 64 | 64 | 32 |
| Discount factor $\gamma$ | 0.99 | 0.99 | 0.99 |
| GAE $\lambda$ | 0.95 | 0.95 | 0.95 |
| Clip range $\epsilon$ | 0.2 | 0.2 | 0.2 |
| Entropy coefficient | 0.01 | 0.01 | 0.01 |
| Value function coefficient | 0.5 | 0.5 | 0.5 |
| Max gradient norm | 0.5 | 0.5 | 0.5 |

The PPO objective for each agent $n$ is:

$$
L^{\text{PPO}}(\theta^n) = \hat{\mathbb{E}}_t \left[ \min\left( \rho_t(\theta^n) \hat{A}_t, \; \text{clip}(\rho_t(\theta^n), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]
$$

where $\rho_t(\theta^n) = \pi_{\theta^n}(a_t | o_t) / \pi_{\theta_{\text{old}}^n}(a_t | o_t)$ is the probability ratio, $\hat{A}_t$ is the Generalized Advantage Estimate (GAE) [28]:

$$
\hat{A}_t = \sum_{l=0}^{T-t-1} (\gamma \lambda)^l \delta_{t+l}, \qquad \delta_t = r_t + \gamma V(o_{t+1}) - V(o_t)
$$

and $\epsilon = 0.2$ is the clipping parameter. The total loss includes the value function loss and an entropy bonus:

$$
L^{\text{total}}(\theta^n) = L^{\text{PPO}}(\theta^n) - c_v \cdot L^{\text{VF}}(\theta^n) + c_e \cdot H[\pi_{\theta^n}]
$$

with $c_v = 0.5$ and $c_e = 0.01$.

GPU-aware training automatically detects CUDA or MPS devices and adjusts batch sizes accordingly (1,024 for central, 512 for microgrid and sub-agents on GPU).

### G. Constraint Handling

Constraint satisfaction is enforced through two complementary mechanisms:

**1) Reward-Based Soft Constraints.** The `GridConstraintValidator` module monitors all physical and regulatory constraints at each timestep and produces a `ConstraintReport` containing normalized severity scores $\sigma_v \in [0, 1]$ for each violation. These severities are integrated into the reward functions as penalty terms (Section III-D), creating negative gradient signals that steer policies away from constraint-violating regions of the action space.

The validator monitors six constraint categories:

| Category | Standard | Limits |
|----------|----------|--------|
| Frequency band | IEGC | 49.5--50.5 Hz |
| UFLS thresholds | IEGC | 49.0 / 48.5 / 48.0 Hz |
| Bus voltage | IEEE | 0.95--1.05 pu |
| Line thermal loading | — | 80% warning, 100% critical |
| Generator ramp rate | — | 10% of $P_{\max}$ per step |
| Generator output limits | — | $P_{\min}$--$P_{\max}$ |

**2) Action-Space Clipping.** All agent actions are clipped to their respective Gymnasium `Box` bounds before application to the physics engine. Central agent actions are clipped to $[-1, 1]$; microgrid and sub-agent actions are clipped to $[0, 1]$. This hard constraint prevents physically infeasible setpoints from being passed to the power flow solver.

### H. Stochastic Operating Conditions

To expose agents to a representative distribution of operating scenarios during training, the simulation employs three stochastic profile generators:

1. **Solar irradiance:** A diurnal bell-curve base profile modulated by an Ornstein-Uhlenbeck (O-U) cloud transient process with mean-reversion.
2. **Wind speed:** An O-U process with configurable mean, volatility, and mean-reversion rate, producing temporally correlated wind speed trajectories.
3. **Load demand:** A diurnal load curve with morning and evening peaks, scaled by a stochastic load factor.

Each episode draws a new random realisation of these profiles, ensuring that the learned policies generalise across a wide range of renewable generation and demand conditions rather than overfitting to a single deterministic scenario.

### I. Evaluation Framework

Eight evaluation scenarios aligned with IEEE and NERC standards have been implemented for post-training assessment:

1. **Base Case:** 24-hour diurnal operation (IEEE 1547-2018 §11.2);
2. **High Renewable Penetration:** 80%+ RE fraction (IEEE 1547-2018 §6.5, IEEE 2800-2022);
3. **N-1 Generator Contingency:** Trip of largest generator (600 MW nuclear) at step 48 (NERC TPL-001-4);
4. **N-1 Tie-Line Contingency:** Trip of tie-line AB at step 72 (NERC TPL-001-4);
5. **Load Ramp:** 20% sudden load increase at step 36 (IEEE 1547-2018 §6.5);
6. **Islanding:** Area C disconnection at step 60, reconnection at step 84 (IEEE 1547-2018 §8.2);
7. **Price Spike:** 3× wholesale price during steps 108--156 (IEEE 2030-2011 §5.3);
8. **Low Inertia:** Three generators tripped at step 24, 40% inertia reduction (ENTSO-E criteria).

Four baseline controllers provide comparative benchmarks: (i) no-control (fixed setpoints), (ii) droop ($R = 5\%$, $f_0 = 50$ Hz), (iii) merit-order economic dispatch, and (iv) PI-based AGC.
