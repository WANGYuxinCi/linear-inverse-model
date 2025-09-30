# Linear Inverse Model (LIM) — Python Toolbox

## Description

### 1. Linear Inverse Model (LIM) overview
The LIM assumes the relevant dynamics can be represented as a linear system forced by stochastic noise (Hasselmann, 1988; Penland & Sardeshmukh, 1995), and written in the form of a linear stochastic differential equation:

<div align="center">

$$
\frac{d\mathbf{𝐱}}{dt}=\mathbf{𝐋𝐱}+\boldsymbol{𝛏}
$$

</div>

Here, 𝐱(t) is the state vector of the system, 𝐋 is the dynamical operator matrix describing the dynamical features of the evolution of, 𝛏 is the stochastic forcing (i.e., white noise)  

The LIM framework has been widely used in climate prediction and predictability studies, including (but not limited to) sea surface temperature, such as ENSO (Penland & Magorian 1993; Shin et al. 2021; Vimont et al. 2022), interannual to decadal climate modes (Newman 2007; Alexander et al. 2008; Lou et al. 2020, 2021), and marine heatwaves (Capotondi et al. 2022; Xu et al. 2022; Wang et al. 2023), subsurface ocean temperature (Xu et al. 2024), sea surface height (Shin & Newman 2017), precipitation (Devanand et al. 2024) and sea-ice (Brennan et al. 2023).

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

## Description

The linear inverse model toolbox can be installed by:

#### From GitHub
```bash
pip install git+https://github.com/WANGYuxinCi/linear-inverse-model.git
```

## Quick Start

**Wyrtki_CSLIM_cookbook.ipynb** is a Jupyter notebook that reproduces the "Wyrtki-CSLIM" used for ENSO forecasting in Wang et al. (submitted), using `CSLIM.py`. It also showcases how to initialize, train, and run deterministic forecasts with CSLIM. The STLIM workflow, implemented in `STLIM.py`, is analogous to the CSLIM example.

If you encounter problems in running the linear inverse model or have questions, please feel free to contact Yuxin Wang (yuxinw@hawaii.edu).

---

### References

Alexander, M. A., L. Matrosova, C. Penland, J. D. Scott, and P. Chang, 2008: Forecasting Pacific SSTs: Linear Inverse Model Predictions of the PDO. J. Climate, 21, 385–402, https://doi.org/10.1175/2007JCLI1849.1.

Capotondi, A., M. Newman, T. Xu, and E. Di Lorenzo, 2022: An optimal precursor of northeast Pacific marine heatwaves and central Pacific El Niño events. Geophys. Res. Lett., 49, e2021GL097350, https://doi.org/10.1029/2021GL097350.

Devanand, A., and Coauthors, 2024: Australia’s Tinderbox Drought: An extreme natural event likely worsened by human-caused climate change. Sci. Adv., 10, eadj3460, https://doi.org/10.1126/sciadv.adj3460.

Hasselmann, K., 1988: PIPs and POPs: The reduction of complex dynamical systems using principal interaction and oscillation patterns. J. Geophys. Res., 93, 11 015–11 021, https://doi.org/10.1029/JD093iD09p11015.

Lou, J., T. J. O’Kane, and N. J. Holbrook, 2020: A Linear Inverse Model of Tropical and South Pacific Seasonal Predictability. J. Climate, 33, 4537–4554, https://doi.org/10.1175/JCLI-D-19-0548.1.

Lou, J., T. J. O’Kane, and N. J. Holbrook, 2021: A Linear Inverse Model of Tropical and South Pacific Climate Variability: Optimal Structure and Stochastic Forcing. J. Climate, 34, 143–155, https://doi.org/10.1175/JCLI-D-19-0964.1.

Newman, M., 2007: Interannual to decadal predictability of tropical and North Pacific sea surface temperatures. J. Climate, 20, 2333–2356, https://doi.org/10.1175/JCLI4165.1.

Penland, C., and T. Magorian, 1993: Prediction of Niño 3 Sea Surface Temperatures Using Linear Inverse Modeling. J. Climate, 6, 1067–1076, https://doi.org/10.1175/1520-0442(1993)006<1067:PONSST>2.0.CO;2.

Penland, C., and P. D. Sardeshmukh, 1995: The optimal growth of tropical sea surface temperature anomalies. J. Climate, 8, 1999–2024, https://doi.org/10.1175/1520-0442(1995)008<1999:TOGOTS>2.0.CO;2.

Shin, S.-I., and M. Newman, 2021: Seasonal Predictability of global and North American coastal sea surface temperature and height anomalies. Geophys. Res. Lett., 48, e2020GL091886, https://doi.org/10.1029/2020GL091886.

Shin, S. I., P. D. Sardeshmukh, M. Newman, C. Penland, and M. A. Alexander, 2021: Impact of annual cycle on ENSO variability and predictability. J. Climate, 34, 171–193, https://doi.org/10.1175/JCLI-D-20-0291.1.

Vimont, D. J., M. Newman, D. S. Battisti, and S. Shin, 2022: The Role of Seasonality and the ENSO Mode in Central and East Pacific ENSO Growth and Evolution. J. Climate, 35, 3195–3209, https://doi.org/10.1175/JCLI-D-21-0599.1.

Wang, Y., N. J. Holbrook, and J. B. Kajtar, 2023: Predictability of Marine Heatwaves off Western Australia Using a Linear Inverse Model. J. Climate, 36, 6177–6193, https://doi.org/10.1175/JCLI-D-22-0692.1.

Xu, T., M. Newman, A. Capotondi, S. Stevenson, E. Di Lorenzo, and M. A. Alexander, 2022: An increase in marine heatwaves without significant changes in surface ocean temperature variability. Nat. Commun., 13, 7396, https://doi.org/10.1038/s41467-022-34934-x.

Xu, T., M. Newman, M. A. Alexander, and A. Capotondi, 2024: Seasonal predictability of bottom temperatures along the North American West Coast. J. Geophys. Res. Oceans, 129, e2023JC020504, https://doi.org/10.1029/2023JC020504
