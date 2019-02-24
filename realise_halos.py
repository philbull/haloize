#!/usr/bin/env python
"""
Realise a set of halos in voxels.
"""
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import pylab as plt
import pyccl as ccl

C_kms = 3e5 # km/s
fsky = 1.
MH_MIN, MH_MAX, MH_BINS = 1e12, 1e16, 200



class HaloFactory(object):
    
    def __init__(self):
        """
        Initialise.
        """
        Mh = Mh_range(z, **args)
        dndlog10m = ccl.massfunc(cosmo, Mh, 1./(1.+z))
        
        
    def x(self):
        pass




# Specify cosmology
cosmo = ccl.Cosmology(h=0.67, Omega_c=0.25, Omega_b=0.045, n_s=0.965, sigma8=0.834)

z = 0.

# Halo mass function
Mh = np.logspace(np.log10(MH_MIN), np.log10(MH_MAX), MH_BINS)
dndlog10m = ccl.massfunc(cosmo, Mh, 1./(1.+z))
bh = ccl.halo_bias(cosmo, Mh, a)

# Cumulative integral of halo mass function
nm = integrate.cumtrapz(dndlog10m[::-1], -np.log10(Mh)[::-1], initial=0.)[::-1]

# Rescale nm to get cdf
cdf = nm / np.max(nm)

# Build interpolator
Mh_interp = interpolate.interp1d(cdf, np.log10(Mh), kind='linear')


np.random.seed(10)
u = np.random.uniform(size=int(1e6))

# Realise halo mass distribution
mh_real = 10.**(Mh_interp(u))


plt.subplot(111)

plt.hist(np.log10(mh_real), bins=100, normed=True, label="%1.0e halos" % u.size)
#plt.plot(np.log10(Mh), cdf*7e3, 'r-', lw=1.8)
plt.plot(np.log10(Mh), dndlog10m/np.max(nm), 'b-', lw=1.8, label="$dn/d\log_{10}M_h$")

plt.xlabel(r"$\log_{10} M_h$ $M_\odot$", size=18)
plt.yscale('log')
plt.legend(loc='upper right', frameon=False)

plt.tight_layout()
plt.show()













exit()


def Mmin(z):
    """
    Minimum halo mass as a function of redshift.
    """
    return MH_MIN + z*0.

def Mh_range(z, mhmin):
    Mmin = mhmin + z*0.
    return np.logspace(np.log10(Mmin), np.log10(MH_MAX), MH_BINS)

def nm(z, **args):
    """
    Calculate number density above some threshold mass, n(>M_min), for each z
    """
    def nm_integ(z):
        """
        Integral to get n(>M_min).
        """
        Mh = Mh_range(z, **args)
        dndlog10m = ccl.massfunc(cosmo, Mh, 1./(1.+z))
        
        # Integrate over mass range
        return integrate.simps(dndlog10m, np.log10(Mh))
    
    # Calculate n(>M_min) as a function of z
    nm_array = np.array([nm_integ(_z) for _z in z])
    return nm_array # shape (N_z,)
    
def Ntot(z, mhmin, fsky=1.):
    """
    Calculate the total number of dark matter halos above a given mass 
    threshold, as a function of maximum redshift and sky fraction.
    """
    # Calculate cumulative number density n(>M_min) as a fn of M_min and redshift 
    ndens = nm(z, mhmin=mhmin)

    # Integrate over comoving volume of lightcone
    r = ccl.comoving_radial_distance(cosmo, a) # Comoving distance, r
    H = 100.*cosmo['h'] * ccl.h_over_h0(cosmo, a) # H(z) in km/s/Mpc
    Ntot = integrate.cumtrapz(ndens * r**2. / H, z, initial=0.)
    Ntot *= 4.*np.pi * fsky * C_kms
    return Ntot


# Specify cosmology
cosmo = ccl.Cosmology(h=0.67, Omega_c=0.25, Omega_b=0.045, n_s=0.965, sigma8=0.834)

# Scale factor array
a = np.linspace(1., 0.2, 500)
z = 1./a - 1.

# Plot Ntot as a function of z_max
plt.subplot(121)
plt.plot(z, Ntot(z, mhmin=1e12), 'k-', lw=1.8, label="$M_h > 10^{12} M_\odot$")
plt.plot(z, Ntot(z, mhmin=5e12), 'k-', lw=1.8, label=r"$M_h > 5 \times 10^{12} M_\odot$", alpha=0.5)
plt.plot(z, Ntot(z, mhmin=1e13), 'r-', lw=1.8, label="$M_h > 10^{13} M_\odot$")
plt.plot(z, Ntot(z, mhmin=1e14), 'b-', lw=1.8, label="$M_h > 10^{14} M_\odot$")

plt.ylim((1e4, 1e10))
plt.yscale('log')
plt.ylabel(r"$N_h(>M_{\rm min})$", size=18)
plt.xlabel(r"$z_{\rm max}$", size=18)

plt.legend(loc='upper left')


# Plot no. density as a function of redshift, and cumulative comoving volume
plt.subplot(122)

# Comoving volume
print("h=", cosmo['h'])
r = ccl.comoving_radial_distance(cosmo, a) # Comoving distance, r
H = 100.*cosmo['h'] * ccl.h_over_h0(cosmo, a) # H(z) in km/s/Mpc
vol = integrate.cumtrapz(r**2. / H, z, initial=0.)
vol *= 4.*np.pi * fsky * C_kms

plt.plot(z, nm(z, mhmin=1e12), 'k-', label=r"$n(>M_{\rm min}, z)$ $[{\rm Mpc}^{-3}]$")
plt.plot(z, vol, 'r-', label=r"$V(z)$ $[{\rm Mpc}^3]$")
plt.plot(z, r, 'b-', label=r"$r(z)$ $[{\rm Mpc}]$")

plt.xlabel(r"$z_{\rm max}$", size=18)

plt.legend()
plt.yscale('log')

plt.tight_layout()
plt.show()

exit()


# Plot n(>M_min)
for i in [10, 100, 150]:
    plt.plot(z, ndens[:,i], lw=1.8, label="Mh_min = %3.3e" % mh[i])
#plt.plot(z, ndens[:,100], 'g-', lw=1.8)
#plt.plot(z, ndens[:,150], 'b-', lw=1.8)
plt.yscale('log')
plt.xlabel("z")
plt.ylabel(r"$n(>M_{\rm min})$ $[{\rm Mpc}^{-3}]$")

plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


#ccl.massfunc(cosmo, halo_mass, a, overdensity=200)


# Halo mass function, dn/dlog_10M
#dndlogm = ccl.massfunc(cosmo, Mh, a)
