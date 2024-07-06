import numpy as np
import pandas as pd
from dustapprox.models import PrecomputedModel

lib = PrecomputedModel()

def kV():
    ## return Av/A0 for Johnson V band
    ## We work with Av, but the model uses A0. This function gives a quick and dirty conversion factor
    filename = lib.find(passband='Johnson')[0]['filename']
    model = lib.load_model(filename)
    modelV = model[2].to_pandas()
    ## X is A0 (monochromatic extinction at 550 nm)
    ## Y is Teff, normalized to 5040K
    const = modelV['1'][0]
    x1 = modelV['A0'][0]
    y1 = modelV['teffnorm'][0]
    x2 = modelV['A0^2'][0]
    y2 = modelV['teffnorm^2'][0]
    xy = modelV['A0 teffnorm'][0]
    x2y = modelV['A0^2 teffnorm'][0]
    xy2 = modelV['A0 teffnorm^2'][0]
    x3 = modelV['A0^3'][0]
    y3 = modelV['teffnorm^3'][0]
    km_poly= lambda x,y: const + x*x1 + y*y1 + x*x2 + y*y2 + x*y*xy + x*x2 + y*x2y + x*y2*xy2 + x*x3 + y*y3
    x = np.arange(0.01,20,0.01)
    y = np.arange(3500,10000,250)/5040
    X,Y = np.meshgrid(x,y)
    Z = km_poly(X,Y)
    return Z.mean(), Z.std()

const_kV = 1.02 ## Av = A0 * const_kV: rough but convenient conversion factor

def get_kX(Av,teff,model):
    ## returns Av/Ax for a given Av, teff and model
    ## we calculate kX through k0X = Ax/A0, for which the model is available
    c0 = model['1'].iloc[0]
    cx = model['A0'].iloc[0]
    cy = model['teffnorm'].iloc[0]
    cxx = model['A0^2'].iloc[0]
    cyy = model['teffnorm^2'].iloc[0]
    cxy = model['A0 teffnorm'].iloc[0]
    cxxy = model['A0^2 teffnorm'].iloc[0]
    cxyy = model['A0 teffnorm^2'].iloc[0]
    cxxx = model['A0^3'].iloc[0]
    cyyy = model['teffnorm^3'].iloc[0]
    x = Av / const_kV ## kV = Av/A0 ---->  A0 = Av/kV
    y = teff/5040

    k0X = c0 + cx*x + cy*y + cxx*x*2 + cyy*y*2 + cxy*x*y + cxxy*x**2*y + cxyy*x*y**2 + cxxx*x**3 + cyyy*y**3 ## k0X = Ax/A0
    kX = k0X * const_kV  ## kX = Ax/Av
    return kX

def get_Gaia_extinction(Av,bprp0=0,teff=None):
    # returns Ag, Abp, Arp for a given Av and teff
    filename = lib.find(passband='Gaia')[0]['filename']
    model = lib.load_model(filename)
    modelG,modelBP,modelRP = model[0].to_pandas(),model[1].to_pandas(),model[2].to_pandas()
    if teff is None:
        teff = get_teff(bprp0)
    kG = get_kX(Av,teff,model=modelG) 
    kBP = get_kX(Av,teff,model=modelBP)
    kRP = get_kX(Av,teff,model=modelRP)

    return kG * Av, kBP * Av, kRP * Av

def get_AG_EBPRP(Av,bprp0=0,teff=None):
    # returns Ag, E(BP-RP) for a given Av and teff
    if teff is None:
        teff = get_teff(bprp0)
    ag,abp,arp = get_Gaia_extinction(Av,teff)
    return ag,abp-arp
    
def get_Galex_extinction(Av,bprp0=0,teff=None):
    # returns ANUV, AFUV for a given Av, teff
    filename = lib.find(passband='GALEX')[0]['filename']
    model = lib.load_model(filename)
    modelFUV,modelNUV = model[0].to_pandas(),model[1].to_pandas()
    if teff is None:
        teff = get_teff(bprp0)
    kFUV = get_kX(Av,teff,model=modelFUV)
    kNUV = get_kX(Av,teff,model=modelNUV)
    return kNUV * Av, kFUV * Av

def get_Johnson_extinction(Av,bprp0=0,teff=None):
    # returns AU, AB, AV, AR, AI for a given Av, teff
    filename = lib.find(passband='Johnson')[0]['filename']
    model = lib.load_model(filename)
    modelU,modelB,modelV,modelR,modelI = model[0].to_pandas(),model[1].to_pandas(),model[2].to_pandas(),model[3].to_pandas(),model[4].to_pandas()
    if teff is None:
        teff = get_teff(bprp0)
    kU = get_kX(Av,teff,model=modelU) 
    kB = get_kX(Av,teff,model=modelB) 
    kV = get_kX(Av,teff,model=modelV) 
    kR = get_kX(Av,teff,model=modelR) 
    kI = get_kX(Av,teff,model=modelI) 
    return kU * Av, kB * Av, kV * Av, kR * Av, kI * Av

def get_SDSS_extinction(Av,bprp0=0,teff=None):
    # returns u, g, r, i, z for a given Av, teff
    filename = lib.find(passband='SDSS')[0]['filename']
    model = lib.load_model(filename)
    modelu,modelg,modelr,modeli,modelz = model[0].to_pandas(),model[1].to_pandas(),model[2].to_pandas(),model[3].to_pandas(),model[4].to_pandas()
    if teff is None:
        teff = get_teff(bprp0)
    ku = get_kX(Av,teff,model=modelu)
    kg = get_kX(Av,teff,model=modelg)
    kr = get_kX(Av,teff,model=modelr)
    ki = get_kX(Av,teff,model=modeli)
    kz = get_kX(Av,teff,model=modelz)
    return ku * Av, kg * Av, kr * Av, ki * Av, kz * Av

def get_2MASS_extinction(Av,bprp0=0,teff=None):
    # returns J, H, Ks for a given Av, teff
    filename = lib.find(passband='2MASS')[0]['filename']
    model = lib.load_model(filename)
    modelJ,modelH,modelKs = model[0].to_pandas(),model[1].to_pandas(),model[2].to_pandas()
    if teff is None:
        teff = get_teff(bprp0)
    kJ = get_kX(Av,teff,model=modelJ) 
    kH = get_kX(Av,teff,model=modelH) 
    kKs = get_kX(Av,teff,model=modelKs)
    return kJ * Av, kH * Av, kKs * Av

def get_WISE_extinction(Av,bprp0=0,teff=None):
    # returns W1, W2, W3, W4 for a given Av, teff
    filename = lib.find(passband='WISE')[0]['filename']
    model = lib.load_model(filename)
    modelW1,modelW2,modelW3,modelW4 = model[0].to_pandas(),model[1].to_pandas(),model[2].to_pandas(),model[3].to_pandas()
    if teff is None:
        teff = get_teff(bprp0)
    kW1 = get_kX(Av,teff,model=modelW1) 
    kW2 = get_kX(Av,teff,model=modelW2) 
    kW3 = get_kX(Av,teff,model=modelW3) 
    kW4 = get_kX(Av,teff,model=modelW4) 
    return kW1 * Av, kW2 * Av, kW3 * Av, kW4 * Av

def get_teff(bprp0):
    # returns rough estimate of teff from photometry- to be used in the extinction calculations
    tbl = pd.DataFrame(np.genfromtxt('./zams.dat',names=True,dtype=None,skip_header=13))
    tbl = tbl[tbl['Mini'] < 5]
    bprp = tbl['G_BPmag'] - tbl['G_RPmag']
    t = 10**tbl['logTe']
    t = t[bprp.argsort()]
    bprp = bprp[bprp.argsort()]
    teff = np.interp(bprp0,bprp,t)
    return teff


