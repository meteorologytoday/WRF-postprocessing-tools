import xarray as xr
import pandas as pd
import numpy as np
from shared_constants import *
import scipy.signal

def isIn(pool, *xs):
    if pool is None:
        return True

    for x in xs:
    
   
        if x in pool:
            return True

    return False


def integrateVertically(X, ds, avg=False):

    MUB = ds.MUB
    DNW = ds.DNW
    MU_FULL = ds.MU + ds.MUB
    MU_STAR = MU_FULL / MUB
    integration_factor = - MUB * DNW / g0  # notice that DNW is negative, so we need to multiply by -1

    X_STAR = X * MU_STAR
    X_INT = (integration_factor * X_STAR).sum(dim="bottom_top")

    if avg:
        sum_integration_factor = integration_factor.sum(dim="bottom_top")
        X_INT = X_INT / sum_integration_factor

    return X_INT

def genKernel(size, shape, sigma=1):

    """
    Creates a 2D Gaussian kernel.

    Parameters:
        size: The size of the kernel (should be an odd integer).
        sigma: The standard deviation of the Gaussian distribution.
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    if shape == "gaussian":
        g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2)))
    elif shape == "flat":
        g = np.zeros_like(x) + 1
    else:
        raise Exception("Unknown shape: %s" % (shape,))

    return g / g.sum()


def filterVariable(da, kernel):

    t_len = len(da.coords["time"])
    da_lp = da.copy()
    for t in range(t_len):
        
        tmp = da.isel(time=t).to_numpy()
        tmp_lp = scipy.signal.convolve2d(tmp, kernel, mode='same', boundary='symm')
        
        da_lp[t, :, :] = tmp_lp


    return da_lp


def genAnalysis(
    ds,
    data_interval,
    varnames = None,
):
    
    merge_data = []


    if isIn(varnames, "PH850", "PH500"):

        ds_col = ds.mean(dim=["time", "south_north", "west_east"])
        base_pressure = ds_col.PB.to_numpy()
        ps = [850, 500] # hPa
        zidxs = []
        for p in ps:
            zidxs.append(
                np.argmin(np.abs(base_pressure - p*100))
            )

            print("Found constant pressure surface for p=%d hPa => zidx=%d. Exact number is: %f hPa" % (p, zidxs[-1], base_pressure[zidxs[-1]]/1e2) )


        PH_TTL = (ds.PHB + ds.PH).rename("PH_TTL")

        for var, prefix in [
            (PH_TTL, "PH"),
        ]:
            for i, p in enumerate(ps):
                newname = "%s%d" % (prefix, p,)
                zidx = zidxs[i]
                tmp = var.isel(bottom_top_stag = zidx)
                tmp = tmp.rename(newname)
                print("Add new variable: ", newname)
                merge_data.append(
                    tmp
                )

    if isIn(varnames, "TTL_RAIN", "PRECIP", "TTL_RAIN_LP", "RAINNC_LP", "RAINC_LP"):
        TTL_RAIN = ds["RAINNC"] + ds["RAINC"] #+ ds["RAINSH"] + ds["SNOWNC"] + ds["HAILNC"] + ds["GRAUPELNC"]
        TTL_RAIN = TTL_RAIN.rename("TTL_RAIN")
        merge_data.append(TTL_RAIN)

        PRECIP = ( TTL_RAIN - TTL_RAIN.shift(time=1) ) / (data_interval.total_seconds() / 3600.0) # mm / hr
        #print("PRECIP BEFORE: ", PRECIP)
        #PRECIP = PRECIP.shift(time=-1).rename(time="time_mid").isel(time_mid=slice(0, -1)).drop_vars("XTIME").rename("PRECIP") 
        PRECIP = PRECIP.shift(time=-1).rename(time="time_mid").isel(time_mid=slice(0, -1)).rename("PRECIP").drop_vars("XTIME", errors="ignore")

        #if "XTIME" in PRECIP.keys():
        #    PRECIP = PRECIP.drop_vars("XTIME")
 
        merge_data.append(PRECIP)

        #print("PRECIP: ", PRECIP)
        #print(np.any(np.isfinite(PRECIP.to_numpy())))
        # Low pass
        LP_kernel   = genKernel(5, "flat")
        TTL_RAIN_LP = filterVariable(TTL_RAIN, LP_kernel).rename("TTL_RAIN_LP")
        RAINNC_LP   = filterVariable(ds["RAINNC"], LP_kernel).rename("RAINNC_LP")
        RAINC_LP    = filterVariable(ds["RAINC"], LP_kernel).rename("RAINC_LP")
        merge_data.append(TTL_RAIN_LP)
        merge_data.append(RAINNC_LP)
        merge_data.append(RAINC_LP)
     

    if isIn(varnames, "SST_NOLND"):
       
        # LAND = 1, WATER = 2 
        SST_NOLND = ds["SST"].where(ds["SST"] != 0).rename("SST_NOLND")
        merge_data.append(SST_NOLND)

    
    if isIn(varnames, "IVT", "IVT_x", "IVT_y"):
        # Convert V_T
        V_T = xr.zeros_like(ds["T"]).rename("V_T")
        _tmp = ds["V"].to_numpy()
        V_T[:, :, :, :] = _tmp[:, :, 0:1, :]


        # Convert U_T
        _tmp = ds["U"].to_numpy()
        U_T = xr.zeros_like(ds["T"]).rename("U_T")

        U_T[:, :, :, :] = (_tmp[:, :, :, 1:] + _tmp[:, :, :, :-1]) / 2
        U_T = U_T.rename("U_T")
     
        # IWV
        IWV = integrateVertically(ds["QVAPOR"], ds, avg=False).rename("IWV")
     
        # IVT
        IVT_x = integrateVertically(ds["QVAPOR"] * U_T, ds, avg=False)
        IVT_y = integrateVertically(ds["QVAPOR"] * V_T, ds, avg=False)
        
        # code order matters here
        IVT = ((IVT_x**2 + IVT_y**2)**0.5).rename("IVT")
        IVT_x = IVT_x.rename("IVT_x")
        IVT_y = IVT_y.rename("IVT_y")

        merge_data.extend([IWV, IVT, IVT_x, IVT_y])




    new_ds = xr.merge(merge_data)
    return new_ds

