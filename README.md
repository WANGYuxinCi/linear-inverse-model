# Linear Inverse Model (LIM) — Python Toolbox

## Description

### 1. Linear Inverse Model (LIM) overview
The LIM assumes the relevant dynamics can be represented as a linear system forced by stochastic noise, and written in the form of a linear stochastic differential equation:

<p align="center"><sup>𝑑𝐱</sup>/<sub>𝑑𝑡</sub> = 𝐋𝐱 + 𝛏(t)</p>

Here, 𝐱(t) is the state vector of the system, 𝐋 is the dynamical operator matrix describing the dynamical features of the evolution of, 𝛏(t) is the stochastic forcing (i.e., white noise), and t is time.  

The LIM framework has been widely used in climate prediction and predictability studies, including (but not limited to) sea surface temperature, such as ENSO, other climate modes, and marine heatwaves, subsurface ocean temperature, sea surface height, and precipitation.

### 2. What does this toolbox provide?
This Python toolbox covers common LIM applications, including:

**(1) LIM training**  
  - Estimate the dynamical operator 𝐋, propagation operator 𝐆, and the noise covariance 𝐐  
  - Built-in LIM stability checks
  
**(2) Deterministic forecasts**  
  Make deterministic forecasts by LIM, by propagating an initial state with 𝐆 to produce the future state (omitting noise)
  
**(3) Generate synthetic simulations**  
  Generate synthetic simulations for the state vectors, with consistent dynamical operator 𝐋, noise covariance 𝐐, but random noise forcing
  
**(4) Optimal initial conditions**  
  Calculate the optimal initial conditions (a.k.a optimal initial structure or optimal precursor) for a specified target (final) state, and the corresponding maximum amplification factors
  
**(5) Principal oscillation patterns**  
  Extract damped linear modes, and their periods/damping rates, in the LIM framework

### 3. LIM variants: Stationary vs. Cyclostationary

This toolbox implements both types of LIMs:

- **STationary LIM (STLIM)**: assumes time-invariant operators, i.e., the dynamical operator 𝐋, the propagation operator 𝐆, and noise covariance 𝐐 are always fixed and constant in time. Implemented in `STLIM.py`.
- **CycloStationary LIM (CSLIM)**: allows operators to vary periodically over a known cycle, written as 𝐋<sub>j</sub>, 𝐆<sub>j</sub>, and 𝐐<sub>j</sub> for phase j. For example, in climate studies, a CSLIM trained on monthly data can capture seasonally varying dynamics, and is recommended when the phenomena of interest exhibit strong seasonal features. Implemented in `CSLIM.py`.

---

If you encounter problems in running the linear inverse model or have questions, please feel free to contact Yuxin Wang (yuxinw@hawaii.edu).
