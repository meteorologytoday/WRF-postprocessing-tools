import xarray as xr

g0 = 9.81

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


def integrateVerticallyByPressure(X, ds, avg=False):

    P_FULL = ds.P + ds.PB
    dP = P_FULL - P_FULL.shift()
    integration_factor = - MUB * DNW / g0  # notice that DNW is negative, so we need to multiply by -1

    X_STAR = X * MU_STAR
    X_INT = (integration_factor * X_STAR).sum(dim="bottom_top")

    if avg:
        sum_integration_factor = integration_factor.sum(dim="bottom_top")
        X_INT = X_INT / sum_integration_factor

    return X_INT







