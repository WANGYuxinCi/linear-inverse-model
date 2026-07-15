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


def compute_fixed_climatology_anomaly(
    ds: xr.DataArray,
    time_dim: str = "time",
    base_start: int = 1991,
    base_end: int = 2020,
) -> xr.DataArray:
    """
    Calculate anomalies by removing a fixed monthly climatology.

    This function does not remove a linear or nonlinear trend. It calculates
    one climatological mean for each calendar month using a fixed reference
    period and subtracts that monthly climatology from the entire time series.

    Mathematically:

        anomaly(x, t) = data(x, t) - climatology(x, month(t))

    where the climatology is calculated separately for January, February,
    ..., December over the selected base period.

    For example, when base_start=1991 and base_end=2020:

        January anomaly =
            January value
            - mean of all January values from 1991 through 2020

        February anomaly =
            February value
            - mean of all February values from 1991 through 2020

    The same fixed 1991-2020 monthly climatology is applied to every year,
    including years before 1991 and after 2020.

    Parameters
    ----------
    ds : xr.DataArray
        Input data with a datetime coordinate. The expected dimensions are
        typically:

            (time, lat, lon)

        but additional spatial dimensions are also supported.

    time_dim : str, default="time"
        Name of the time dimension. For the ORAS5 data, use:

            time_dim="time_counter"

    base_start : int, default=1991
        First year of the fixed climatological reference period.

    base_end : int, default=2020
        Last year of the fixed climatological reference period.

    Returns
    -------
    anomaly : xr.DataArray
        Anomaly field with the same dimensions and datetime coordinates as
        the input DataArray.

        No trend is removed. The output is:

            original data - fixed monthly climatology

    Notes
    -----
    This function is appropriate when you want the conventional SST anomaly
    relative to a fixed climatological period, such as 1991-2020.

    If the input is absolute SST, the output is the conventional SST anomaly.
    If the input has already been detrended, the output will instead be the
    climatological anomaly of that detrended field.
    """

    # Make sure that the requested time dimension exists.
    if time_dim not in ds.dims:
        raise ValueError(
            f"`{time_dim}` is not present in dimensions: {ds.dims}"
        )

    # Convert the beginning and ending years into complete date strings.
    # These strings are used to select the fixed climatology period.
    start_date = f"{base_start}-01-01"
    end_date = f"{base_end}-12-31"

    # ---- 1) Select the fixed climatological reference period ----
    #
    # For base_start=1991 and base_end=2020, this selects all available
    # monthly data from January 1991 through December 2020.
    base = ds.sel({
        time_dim: slice(start_date, end_date)
    })

    # Stop with a clear error if the requested climatological period is not
    # present in the input dataset.
    if base.sizes.get(time_dim, 0) == 0:
        raise ValueError(
            f"No data found in climatology period "
            f"[{start_date} through {end_date}]"
        )

    # ---- 2) Calculate the fixed monthly climatology ----
    #
    # This produces 12 climatological maps:
    #
    #     month=1  -> January climatology
    #     month=2  -> February climatology
    #     ...
    #     month=12 -> December climatology
    #
    # Missing values are ignored when calculating the climatological means.
    monthly_climatology = base.groupby(
        f"{time_dim}.month"
    ).mean(
        dim=time_dim,
        skipna=True,
    )

    # ---- 3) Remove the monthly climatology from the complete record ----
    #
    # Each January is compared with the January climatology, each February
    # is compared with the February climatology, and so on.
    anomaly = (
        ds.groupby(f"{time_dim}.month")
        - monthly_climatology
    )

    # The groupby operation may leave an auxiliary "month" coordinate.
    # It is not needed after calculating the anomalies.
    if "month" in anomaly.coords:
        anomaly = anomaly.drop_vars(
            "month",
            errors="ignore",
        )

    # Preserve the original metadata, such as units and variable description.
    anomaly = anomaly.assign_attrs(ds.attrs)

    # Add information describing how the anomaly was calculated.
    anomaly.attrs.update({
        "anomaly_base_period": (
            f"{base_start}-{base_end} fixed monthly climatology"
        ),
        "detrend": "none",
        "anomaly_method": (
            "Original data minus fixed monthly climatology"
        ),
    })

    # Preserve the original variable name.
    anomaly.name = ds.name

    return anomaly


def compute_linear_detrended_fixed_climatology_anomaly(
    ds: xr.DataArray,
    time_dim: str = "time",
    base_start: int = 1991,
    base_end: int = 2020,
) -> xr.DataArray:
    """
    Remove a grid-point linear trend and then remove a fixed monthly
    climatology from the detrended field.

    The calculation follows two main steps:

        1. Fit and remove a linear trend separately at every grid point.
        2. Calculate and remove a fixed monthly climatology from the
           detrended data.

    Mathematically, the fitted linear trend is:

        trend(x, t) = slope(x) * t + intercept(x)

    The detrended field is:

        detrended(x, t) = data(x, t) - trend(x, t)

    The final anomaly is:

        anomaly(x, t) =
            detrended(x, t)
            - climatology_of_detrended_data(x, month(t))

    Therefore, the output is:

        original data
        - grid-point linear trend
        - fixed monthly climatology of the detrended data

    A separate trend is fitted at every spatial grid point. This allows
    different locations to have different warming or cooling rates.

    Parameters
    ----------
    ds : xr.DataArray
        Input data with a datetime coordinate. The expected dimensions are
        typically:

            (time, lat, lon)

        Additional spatial dimensions are also supported.

    time_dim : str, default="time"
        Name of the time dimension. For the ORAS5 data, use:

            time_dim="time_counter"

    base_start : int, default=1991
        First year of the fixed climatological reference period.

    base_end : int, default=2020
        Last year of the fixed climatological reference period.

    Returns
    -------
    anomaly : xr.DataArray
        Linearly detrended fixed-climatology anomaly field with the same
        dimensions and datetime coordinates as the input.

    Notes
    -----
    The trend is fitted and evaluated using the same numerical time index:

        0, 1, 2, ..., N-1

    This is important because the input time coordinate normally contains
    datetime values. Fitting with datetime values and evaluating with
    0, 1, 2, ... would use inconsistent time coordinates and would not
    correctly remove the trend.

    This function removes the complete fitted linear component, including
    both its slope and intercept. The subsequent monthly-climatology removal
    eliminates any constant offset, so the intercept does not affect the
    final anomaly.
    """

    # Make sure that the requested time dimension exists.
    if time_dim not in ds.dims:
        raise ValueError(
            f"`{time_dim}` is not present in dimensions: {ds.dims}"
        )

    # Convert the beginning and ending climatology years into complete
    # date strings.
    start_date = f"{base_start}-01-01"
    end_date = f"{base_end}-12-31"

    # ---- 1) Create a consistent numerical time coordinate ----
    #
    # The numerical time coordinate is:
    #
    #     0, 1, 2, ..., N-1
    #
    # For monthly data, an increase of one represents one monthly time step.
    time_index_values = np.arange(
        ds.sizes[time_dim],
        dtype="float64",
    )

    # Temporarily replace the datetime coordinate with the numerical time
    # index before fitting the trend.
    #
    # The original input DataArray is not modified because assign_coords()
    # returns a new DataArray.
    ds_for_fit = ds.assign_coords({
        time_dim: time_index_values
    })

    # ---- 2) Fit a grid-point linear trend ----
    #
    # xarray.polyfit() fits:
    #
    #     value(t) = slope * t + intercept
    #
    # independently at every spatial grid point.
    #
    # skipna=True allows the fit to ignore missing observations.
    linear_fit = ds_for_fit.polyfit(
        dim=time_dim,
        deg=1,
        skipna=True,
    )

    # Build a numerical time DataArray for evaluating the trend.
    #
    # Its data values are 0, 1, 2, ..., N-1, matching the values used for
    # fitting. Its coordinate labels are the original datetime values, so
    # the calculated trend remains aligned with the original dataset.
    time_index = xr.DataArray(
        time_index_values,
        dims=(time_dim,),
        coords={time_dim: ds[time_dim]},
        name="time_index",
    )

    # ---- 3) Evaluate the fitted trend ----
    #
    # Because the trend is evaluated using the same numerical time values
    # used during fitting, the resulting linear trend is correct.
    linear_trend = xr.polyval(
        time_index,
        linear_fit.polyfit_coefficients,
    )

    # ---- 4) Remove the grid-point linear trend ----
    #
    # Each grid cell has its own fitted slope and intercept.
    detrended = ds - linear_trend

    # ---- 5) Select the fixed climatological reference period ----
    #
    # The climatology must be calculated from the detrended data because
    # this function follows the order:
    #
    #     detrend first -> calculate climatology -> remove climatology
    base = detrended.sel({
        time_dim: slice(start_date, end_date)
    })

    # Stop with a clear error if the requested climatological period is not
    # present in the input dataset.
    if base.sizes.get(time_dim, 0) == 0:
        raise ValueError(
            f"No data found in climatology period "
            f"[{start_date} through {end_date}]"
        )

    # ---- 6) Calculate the fixed monthly climatology ----
    #
    # This produces one climatological detrended field for each calendar
    # month using only the selected base period.
    monthly_climatology = base.groupby(
        f"{time_dim}.month"
    ).mean(
        dim=time_dim,
        skipna=True,
    )

    # ---- 7) Remove the fixed monthly climatology ----
    #
    # Each detrended January field is compared with the detrended January
    # climatology, each February with the February climatology, and so on.
    anomaly = (
        detrended.groupby(f"{time_dim}.month")
        - monthly_climatology
    )

    # Remove the auxiliary "month" coordinate if groupby created it.
    if "month" in anomaly.coords:
        anomaly = anomaly.drop_vars(
            "month",
            errors="ignore",
        )

    # Preserve the original variable attributes.
    anomaly = anomaly.assign_attrs(ds.attrs)

    # Add metadata describing the detrending and climatology procedures.
    anomaly.attrs.update({
        "anomaly_base_period": (
            f"{base_start}-{base_end} fixed monthly climatology"
        ),
        "detrend": (
            "Linear grid-point trend fitted and evaluated using "
            "the same numerical time index"
        ),
        "anomaly_method": (
            "Remove grid-point linear trend, then remove fixed "
            "monthly climatology from detrended data"
        ),
    })

    # Preserve the original variable name.
    anomaly.name = ds.name

    return anomaly

