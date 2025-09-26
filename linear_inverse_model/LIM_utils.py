#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 13:54:15 2025

@author: yuxinwang
"""

import numpy as np
import xarray as xr
from scipy import stats
import statsmodels.api as sm



def compute_gridwise_quadratic_detrended_running_climatology_anomaly(ds, time_dim='time', clim_years=21):
    """
    Compute climate variable anomalies by first applying grid-wise quadratic detrending,
    then subtracting a running n-year monthly climatology.

    Parameters:
    ds : xarray.DataArray
        climate variable DataArray with dimensions (time, lat, lon) and datetime index.
    time_dim : str
        Name of the time dimension (default: 'time').
    clim_years : int
        Width of the running climatology window (default: 21 years).

    Returns:
    ssta : xarray.DataArray
        SST anomalies after quadratic detrending and running monthly climatology removal.
    """
    ds = ds.copy()
    ds_np = ds.values
    time = ds[time_dim]
    t = np.arange(len(time))

    # --- Step 1: Grid-wise quadratic detrending ---
    detrended = ds.copy()
    for ilat in range(ds.lat.size):
        for ilon in range(ds.lon.size):
            y = ds_np[:, ilat, ilon]
            if np.all(np.isnan(y)):
                continue
            valid = ~np.isnan(y)
            if valid.sum() < 3:  # not enough points to fit quadratic
                continue
            coeffs = np.polyfit(t[valid], y[valid], deg=2)
            trend = np.polyval(coeffs, t)
            detrended[:, ilat, ilon] = y - trend

    # --- Step 2: Running monthly climatology subtraction ---
    years = time.dt.year
    months = time.dt.month
    all_years = np.unique(years.values)
    half_window = clim_years // 2
    clim = xr.full_like(detrended, np.nan)

    for year in all_years:
        start_year = max(all_years[0], year - half_window)
        end_year = min(all_years[-1], year + half_window)

        sel = detrended.sel({time_dim: years.isin(np.arange(start_year, end_year + 1))})
        monthly_clim = sel.groupby(f'{time_dim}.month').mean(dim=time_dim)

        year_mask = years == year
        for m in range(1, 13):
            month_mask = (months == m) & year_mask
            clim.loc[month_mask] = monthly_clim.sel(month=m).broadcast_like(detrended.sel({time_dim: month_mask}))

    # Final anomaly
    anomaly = detrended - clim
    return anomaly




def tau_n(autocorrA, autocorrB, delta_t):
    
    """
    
    This function is used to calculate the "integral time scale determining the
    time period required to gain a new degree of freedom" (cf p252 of Davis (1976))
    
    Davis, R.E. (1976), "Predictability of sea surface temperatures and sea level pressure
    anomalies over the North Pacific Ocean", Journal of Physical Oceanography, 6, 249-266.
    
    Parameters
    ----------
    autocorrA: 1d ndarray
        Autocorrelation function for time series A.
    autocorrB: 1d ndarray
        Autocorrelation function for time series B.
    delta_t: float
        Time interval between each of the N observations. If monthly data then delta_t = 1.
        
    Returns
    -----
    taun: float
        Integral time scale determining the time period required to gain a 
        new degree of freedom (cf p252 of Davis (1976))
                      
    """
    
    taun = 0
    for i in range(len(autocorrA)):
        taun += autocorrA[i] * autocorrB[i] * delta_t

    return taun



def t_statistic(A, B, delta_t):
    
    """
    
    This function is used to calculate the t-statistic using an "effective number of degrees of freedom"
    based on the equation for tau_{n} on p252 of ...  
    
    Davis, R.E. (1976), "Predictability of sea surface temperatures and sea level pressure
    anomalies over the North Pacific Ocean", Journal of Physical Oceanography, 6, 249-266.
    
    Parameters
    ----------
    A: 1d ndarray
        Time series A.
    B: 1d ndarray
        Time series B.
    delta_t: float
        Time interval between each of the N observations. If monthly data then delta_t = 1.
        
    Returns
    -----
    r: float
        Correlation coefficient between the two time series
    autocorrA: 1d ndarray
        Autocorrelation function for time series A.
    autocorrB: 1d ndarray
        Autocorrelation function for time series B.
    df_effec: float
        "effective number of degrees of freedom"
    taun: float
        taun (re Davis (1976)) - integral time scale determining the time period
        to gain a new "degree of freedom"
    t_statistic_value: float
        the t-statistic based on equation used in Exercise 16.24 in "Modern Elementary Statistics"
        by J.E. Freund Prentice-Hall, 574pp.
        HOWEVER, the number of degrees of freedom used are NOT "N-2" but rather "N*delta_t/tau_n"!!!!  (cf Davis (1976) p252).
    p_value: float
        p_value (two sided) associated with t_statistic_value
        (i.e., p_value of the correlation coefficient with the effective number of degrees of freedom)
              
    """

    # Remove NaN values by keeping only valid indices where both A and B are not NaN
    valid_indices = ~np.isnan(A) & ~np.isnan(B)
    A_clean = A[valid_indices]
    B_clean = B[valid_indices]
    
    # Check if there are enough valid data points after removing NaNs
    if len(A_clean) < 2:
        raise ValueError("Not enough valid data points after removing NaNs.")
    
    # Calculate the correlation coefficient between the two time series
    r = stats.pearsonr(A_clean, B_clean)[0]

    # Calculate the autocorrelation functions for each time series
    autocorrA_oneside = sm.tsa.acf(A_clean, nlags=len(A_clean) - 1)
    autocorrB_oneside = sm.tsa.acf(B_clean, nlags=len(B_clean) - 1)
        
    autocorrA_oneside_flip = np.flip(autocorrA_oneside[1:])
    autocorrA = np.zeros((len(A_clean) - 1) * 2 + 1)
    autocorrA[0:len(A_clean) - 1] = autocorrA_oneside_flip
    autocorrA[len(A_clean) - 1:] = autocorrA_oneside
    
    autocorrB_oneside_flip = np.flip(autocorrB_oneside[1:])
    autocorrB = np.zeros((len(B_clean) - 1) * 2 + 1)
    autocorrB[0:len(B_clean) - 1] = autocorrB_oneside_flip
    autocorrB[len(B_clean) - 1:] = autocorrB_oneside

    # Calculate the integral time scale determining the time period required to gain a new "degree of freedom"
    taun = tau_n(autocorrA, autocorrB, delta_t)

    # Calculate the effective number of degrees of freedom
    N = len(A_clean)
    df_effec = N * delta_t / taun

    # Calculate the t-statistic using the effective number of degrees of freedom
    if r == 1:
        print('CORRELATION COEFFICIENT = 1. WILL GET DIVIDE BY ZERO IN T-STATISTIC!!!')

    Nminus2 = df_effec
    t_statistic_value = (r * np.sqrt(Nminus2)) / np.sqrt(1 - r**2)
    
    # Calculate the p-value using the t-statistic and the effective number of degrees of freedom
    p_value = stats.t.sf(np.abs(t_statistic_value), df=df_effec) * 2  # Two-tailed p-value

    return r, autocorrA, autocorrB, df_effec, taun, t_statistic_value, p_value


def calculate_rms(values):
    """
    Calculate the root mean square (RMS) of a list of values, ignoring NaN values.

    Parameters:
    - values (array-like): List or array of numeric values.

    Returns:
    - float: Root mean square of the input values, ignoring NaNs.
    """
    # Convert input to a numpy array if it's not already
    values = np.array(values, dtype=float)
    # Remove NaN values
    valid_values = values[~np.isnan(values)]
    # Check if there are enough valid data points
    if len(valid_values) == 0:
        raise ValueError("Input contains only NaN values or is empty. Please provide valid numeric values.")
    # Calculate the sum of squares
    sum_of_squares = np.sum(valid_values ** 2)
    # Calculate the mean square
    mean_square = sum_of_squares / len(valid_values)
    # Calculate the root mean square
    rms = np.sqrt(mean_square)

    return rms


def area_mean(da, lat_bounds, lon_bounds):
    """Area average with latitude weights."""
    sub = da.sel(
        lat=slice(*sorted(lat_bounds)),
        lon=slice(*sorted(lon_bounds))
    )
    weights = np.cos(np.deg2rad(sub.lat))
    return (sub * weights).mean(dim=['lat', 'lon'])


def flatten_mask_weight(data3d, lat1d, mask=None, weight="sqrtcos"):
    """
    Flatten (time, lat, lon) data to (time, space) with optional ocean mask
    and latitude weighting.

    Parameters
    ----------
    data3d : numpy.ndarray or xarray.DataArray
        Anomaly field with shape (T, Ny, Nx).
        Notes: If provided as an xarray.DataArray, it will be automatically
        converted to a NumPy array.
    lat1d : np.ndarray
        1-D latitude array (Ny,).
    mask : np.ndarray or None, optional
        Boolean mask (Ny, Nx). True=keep, False=drop. If None, inferred from
        the first time slice: ~np.isnan(data3d[0]).
    weight : {None, "cos", "sqrtcos"}, optional
        Latitude weights applied to each grid point after masking:
        - None      : no weighting
        - "cos"     : cos(lat)
        - "sqrtcos" : sqrt(cos(lat))  (common for geophysical EOFs)

    Returns
    -------
    X : np.ndarray
        Flattened, masked, weighted data3d (T, M),
        where M is the number of kept points.
    W : np.ndarray
        Flattened, masked, weighted array (T, M),
        where M is the number of kept points.
    mask2d : np.ndarray
        Boolean mask used (Ny, Nx).
    """
    # Ensure NumPy array
    if isinstance(data3d, xr.DataArray):
        data3d = data3d.values            
    if data3d.ndim != 3:
        raise ValueError("data3d must have shape (time, lat, lon)")
    T, Ny, Nx = data3d.shape
    if lat1d.ndim != 1 or lat1d.size != Ny:
        raise ValueError("lat1d must be 1-D with length equal to data3d.shape[1]")

    # Flatten
    X = data3d.reshape(T, Ny * Nx)

    # Mask
    if mask is None:
        # Require grid cell to be valid at ALL timesteps
        mask2d = np.all(np.isfinite(data3d), axis=0)   # (Ny, Nx)
    else:
        if mask.shape != (Ny, Nx):
            raise ValueError("mask must have shape (lat, lon)")
        mask2d = mask
    keep = mask2d.ravel()
    X = X[:, keep]

    # Weights
    if weight is not None:
        coslat = np.cos(np.deg2rad(lat1d))
        if weight == "sqrtcos":
            wlat = np.sqrt(coslat)
        elif weight == "cos":
            wlat = coslat
        else:
            raise ValueError("weight must be None, 'cos', or 'sqrtcos'")
        W = np.broadcast_to(wlat[:, None], (Ny, Nx)).reshape(Ny * Nx)
        W = W[keep]
        X = X * W

    return X, W, mask2d


def EOF(data, lat1d=None, n_modes=10, mask=None, weight="sqrtcos"):
    """
    EOF analysis via SVD of the (time x space) matrix.

    Supports both:
      - 3D anomalies (T, Ny, Nx): calls flatten_mask_weight for masking/weights.
      - 2D anomalies (T, M): assumes already flattened and weighted anomalies (time, space).

    Parameters
    ----------
    data : np.ndarray
        Anomalies with shape (T, Ny, Nx) or (T, M).
    lat1d : np.ndarray or None
        Required only for 3D input when using latitude-based weights.
    n_modes : int
        Number of leading EOF modes to retain.
    mask : np.ndarray or None
        3D input: boolean (Ny, Nx). 2D input: boolean (M,).
    weight : {None, "cos", "sqrtcos"}, optional
        Latitude weighting (3D only). Ignored if data is 2D.

    Returns
    -------
    pcs : np.ndarray
        (T, n_modes) principal components.
    eofs : np.ndarray
        (n_modes, M_kept) EOF patterns over kept space (rows are modes).
    fve : np.ndarray
        (n_modes,) fraction of variance explained.
    mask_used : np.ndarray
        The mask actually used.
        - 3D input: boolean (Ny, Nx).
        - 2D input: boolean (M,).
    """
    if data.ndim == 3:
        X, W, mask_used = flatten_mask_weight(data, lat1d, mask=mask, weight=weight)
    elif data.ndim == 2:
        X = data.copy()
        if mask is None:
            mask_used = ~np.isnan(X[0])
        else:
            mask_used = np.asarray(mask)
            if mask_used.ndim != 1 or mask_used.size != X.shape[1]:
                raise ValueError("For 2D input, mask must be 1-D of length M.")
        X = X[:, mask_used]
    else:
        raise ValueError("data must be 2D (time, space) or 3D (time, lat, lon)")

    # SVD
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    if n_modes > s.size:
        raise ValueError(f"n_modes={n_modes} exceeds available modes {s.size}")
    
    eofs_full = Vt   # shape (modes, space)
    lam = s**2
    fve_full = lam / np.sum(lam)
    pcs_full = X @ eofs_full.T

    pcs  = pcs_full[:, :n_modes]
    eofs = eofs_full[:n_modes, :]
    fve  = fve_full[:n_modes]

    return pcs, eofs, fve, mask_used


def project_onto_eofs(data3d, eofs, mask2d, lat1d=None, weight="sqrtcos"):
    """
    Project a new anomaly segment onto previously fitted EOFs.

    Parameters
    ----------
    data3d : np.ndarray
        New anomalies to project.
        - If 3D: shape (T_new, Ny, Nx). Will be flattened with the SAME mask/weights
          used during fitting (via `flatten_mask_weight`).
        - If 2D: shape (T_new, M). Assumed already flattened/weighted exactly
          like the training X.
    eofs : np.ndarray
        EOF patterns from EOF() with shape (n_modes, M_kept).
    mask2d : np.ndarray
        Boolean mask used during fitting (Ny, Nx) if 3D input.
    lat1d : np.ndarray, optional
        Required for 3D input when using latitude-based weights.
    weight : {None, "cos", "sqrtcos"}, optional
        Latitude weighting (3D only).

    Returns
    -------
    pcs_new : np.ndarray
        Projected principal components for the new segment (T_new, n_modes).
    """
    if data3d.ndim == 3:
        if lat1d is None and weight in ("cos", "sqrtcos"):
            raise ValueError("lat1d is required for latitude-based weights with 3D input.")
        X_new = flatten_mask_weight(data3d, lat1d, mask=mask2d, weight=weight)[0]
    elif data3d.ndim == 2:
        X_new = data3d
    else:
        raise ValueError("data3d must be 2D (time, space) or 3D (time, lat, lon).")

    if X_new.shape[1] != eofs.shape[1]:
        raise ValueError(
            f"Space mismatch: X_new has {X_new.shape[1]} columns, "
            f"but EOFs expect {eofs.shape[1]}"
        )

    pcs_new = X_new @ eofs.T
    return pcs_new


def rebuild_eof_on_grid(eof_row, mask2d):
    """
    Rebuild a 1-D EOF vector back to a 2-D (lat, lon) grid.

    Parameters
    ----------
    eof_row : np.ndarray
        EOF coefficients over kept points (M,).
    mask2d : np.ndarray
        Boolean mask used during fitting (Ny, Nx).

    Returns
    -------
    eof2d : np.ndarray
        2-D array (Ny, Nx) with NaN on masked points and EOF values elsewhere.
    """
    Ny, Nx = mask2d.shape
    eof2d = np.full((Ny, Nx), np.nan, dtype=float)
    eof2d[mask2d] = eof_row
    return eof2d


def compute_linear_detrended_fixed_climatology_anomaly(
    ds: xr.DataArray,
    time_dim: str = "time",
    base_start: int | str = 1991,         # year (1991) or date "1991-01-01"
    base_end: int | str | None = None,    # year (2020) or date "2020-12-31"
    base_years: int = 30,                 # used when base_end is None
) -> xr.DataArray:
    """
    Anomalies = (grid-wise linear-detrended field) - (fixed monthly climatology over base window).

    Parameters
    ----------
    ds : xr.DataArray  (time, lat, lon, ...)
    time_dim : str
    base_start : int|str
        Base period start (year or ISO date). E.g., 1991 or "1991-01-01".
    base_end : int|str|None
        Base period end (year or ISO date). If None, it's computed as base_start + base_years - 1 (to Dec 31).
    base_years : int
        Window length if base_end is None (default 30).

    Returns
    -------
    anomaly : xr.DataArray  (same shape as ds)
    """

    if time_dim not in ds.dims:
        raise ValueError(f"`{time_dim}` not in dims: {ds.dims}")

    # ---- Normalize base window ----
    def _year_from(x):
        return int(str(x)[:4])

    by = _year_from(base_start)
    if base_end is None:
        ey = by + base_years - 1
        base_start_str = f"{by}-01-01"
        base_end_str   = f"{ey}-12-31"
    else:
        ey = _year_from(base_end)
        # Respect provided full dates if given
        base_start_str = base_start if isinstance(base_start, str) and "-" in str(base_start) else f"{by}-01-01"
        base_end_str   = base_end   if isinstance(base_end,   str) and "-" in str(base_end)   else f"{ey}-12-31"

    # ---- 1) Linear detrend (grid-wise) ----
    t = xr.DataArray(
        np.arange(ds.sizes[time_dim], dtype="float64"),
        dims=(time_dim,), coords={time_dim: ds[time_dim]}, name="t_index",
    )
    pf = ds.polyfit(dim=time_dim, deg=1, skipna=True)          # returns ...polyfit_coefficients(degree)
    trend = xr.polyval(t, pf.polyfit_coefficients)             # a*t + b
    detrended = (ds - trend).assign_attrs(ds.attrs)
    detrended.name = (ds.name + "_detrended") if ds.name else "detrended"

    # ---- 2) Fixed monthly climatology over [base_start, base_end] ----
    base = detrended.sel({time_dim: slice(base_start_str, base_end_str)})
    if base.sizes.get(time_dim, 0) == 0:
        raise ValueError(f"No data in climatology window [{base_start_str} .. {base_end_str}]")

    clim = base.groupby(f"{time_dim}.month").mean(dim=time_dim, skipna=True)

    # Subtract fixed monthly climatology from all times
    anomaly = detrended.groupby(f"{time_dim}.month") - clim
    # Clean up the temporary 'month' coord if it appears
    if "month" in getattr(anomaly, "coords", {}):
        anomaly = anomaly.drop_vars("month", errors="ignore")

    # ---- attrs ----
    anomaly = anomaly.assign_attrs(detrended.attrs)
    anomaly.attrs.update({
        "anomaly_base_period": f"{by}–{ey} (fixed monthly climatology)",
        "detrend": "linear, grid-wise (xarray.polyfit deg=1)",
    })
    anomaly.name = ds.name

    return anomaly




