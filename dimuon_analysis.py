"""
Dimuon Reconstruction Particle Analysis.

This program aims to reconstruct particle resonances from dimuon collision data
from CERN's Open Data Portal using invariant mass calculations from special 
relativity.

The program is split into 5 sections:
1. Reconstruct invariant mass of muon pairs using four-momenta and identify 
   signal region for either the Z boson or Higgs boson.
2. Fit the data to a model of a Gaussian signal and polynomial (quadratic)
   background.
3. Calculate the statistical significance of the signal.
4. Evaluate goodness-of-fit of the model using chi-squared statistics.
5. Produce plots of the data and model with Poisson error bars and residuals.

Particles analysed:
    - Z boson (≈ 91 GeV)
""" 

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def histogram(region, bins=50):
    """
    Histograms the invariant mass data, computing histogram counts, bin edges 
    and bin centres.
    
    Parameters:
    region (ndarray): array of mass values within selected signal region.
    bins (int): number of bins used for histogram.
    
    Returns:
    counts (ndarray): number of events in each bin.
    bin_edges (ndarray): edges of the histogram bins.
    bin_centres (ndarray): centres of the histogram bins.
    """
    counts, bin_edges = np.histogram(region, bins=bins)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    return counts, bin_edges, bin_centres

def load_data():
    """
    The four-momentum components of two muons from each event are read from the dataset.
    Their combined four-momentum is used to compute the invariant mass of each
    event.
    The user-selected mass interval is applied to observe the expected signal.
    
    Returns:
    mass (ndarray): invariant mass of each event.
    region (ndarray): array of mass values within selected signal region.
    """
    data = np.genfromtxt("dimuon_data.txt", delimiter=",", names=True)

    E1 = data["E1"]
    px1 = data["px1"]
    py1 = data["py1"]
    pz1 = data["pz1"]
    E2 = data["E2"]
    px2 = data["px2"]
    py2 = data["py2"]
    pz2 = data["pz2"]
    
    E = E1 + E2
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    
    mass_squared = E**2 - px**2 - py**2 - pz**2
    mass = np.sqrt(np.maximum(mass_squared, 0))
    
    region = mass[(mass > 70) & (mass < 110)]
    
    return mass, region

def signal(x, A, mu, sigma):
    """
    Returns the Gaussian approximation of the signal.
    """
    return A * np.exp(-0.5*((x - mu) / sigma)**2)

def background(x, a, b, c):
    """
    Returns the quadratic approximation of the background.
    """
    return a + b*x + c*x**2

def signal_plus_background(x, A, mu, sigma, a, b, c):
    """
    Returns the combined approximation of the signal and background.
    """
    return signal(x, A, mu, sigma) + background(x, a, b, c)

def fit_signal_background(region):
    """
    Fits a signal + background model to the invariant mass distribution around the
    expected signal.
    The invariant mass values are histogrammed. A combined model of a Gaussian
    signal and quadratic background is fitted using non-linear least squares
    optimisation.
    The Gaussian component approximates the resonance peak of the particle, while
    the quadratic component models the noise from other background processes.
    Initial parameters are estimated from the histogram before optimisation.
    
    Parameters:
    region (ndarray): array of mass values within selected signal region.

    Returns:
    mass_peak (str): Estimated peak mass of the particle with uncertainty.
    mass_width (str): Estimated width of Gaussian with uncertainty.
    mass_FWHM (str): Full width at half maximum of Gaussian, from relation: FWHM = 2*sqrt(2*ln2)*sigma
    popt (ndarray): optimised parameters of signal + background model [A, mu, sigma, a, b, c].
    """
    counts, bin_edges, bin_centres = histogram(region)
    
    A_guess = counts.max()
    mu_guess = bin_centres[counts.argmax()]
    sig_guess = 3
    
    a_guess = np.min(counts)
    b_guess = 0
    c_guess = 0
    
    popt, pcov = curve_fit(signal_plus_background, bin_centres, counts, p0=[A_guess, mu_guess, sig_guess, a_guess, b_guess, c_guess], maxfev=100000)
    A_fit, mu_fit, sigma_fit, a_fit, b_fit, c_fit = popt
    perr = np.sqrt(np.diag(pcov))

    mass_peak = f"{mu_fit:.4f} ± {perr[1]:.4f}"
    mass_width = f"{sigma_fit:.4f} ± {perr[2]:.4f}"
    mass_FWHM = f"{2.355 * sigma_fit:.4f}"
    
    return mass_peak, mass_width, mass_FWHM, popt

def plot_data(mass):
    """
    Plots a histogram of the invariant mass against number of events to
    show the energies at which the most frequent events happened (potential particles).
    
    Parameters:
    mass (ndarray): invariant mass of each event.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(mass, bins=120, range=(0,150))
    ax.set_xlabel("Invariant Mass (GeV)")
    ax.set_ylabel("Events")
    ax.set_title("Reconstructed Dimuon Mass Spectrum (100k events)")
    plt.show()

def plot_fit(region, popt):
    """
    Plots the smooth approximated curve of the data, along with the separation
    into the signal and background, and the raw data along with its error bars. 
    Another graph is then added on the bottom of the residuals to show how good
    the curve approximation of the data is.
    
    Parameters:
    region (ndarray): array of mass values within selected signal region.
    popt (ndarray): optimised parameters of signal + background model [A, mu, sigma, a, b, c].
    """
    A, mu, sigma, a, b, c = popt
    xmin = region.min()
    xmax = region.max()
    x = np.linspace(xmin, xmax, 500)
    counts, bin_edges, bin_centres = histogram(region)
    poisson_errors = np.sqrt(counts)
    
    model_counts = signal_plus_background(bin_centres, *popt)
    residuals = counts - model_counts

    fig = plt.figure(figsize=(7,6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3,1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    
    ax1.errorbar(bin_centres, counts, yerr=poisson_errors, fmt='o', color='blue', label="Data")
    ax1.plot(x, signal_plus_background(x, *popt), label='Signal + background fit', color = "red")
    ax1.plot(x, signal(x, A, mu, sigma), label="Signal", linestyle="--", color = "black")
    ax1.plot(x, background(x, a, b, c), label="Background", linestyle="--", color = "pink")
    ax1.set_xlabel("Invariant Mass (GeV)")
    ax1.set_ylabel("Events")
    ax1.set_title("Z Boson Region")
    ax1.legend()
    
    ax2.axhline(0, color="black")
    ax2.scatter(bin_centres, residuals)
    ax2.set_xlabel("Invariant Mass (GeV)")
    ax2.set_ylabel("Data - Fit")

    plt.tight_layout()
    plt.show()

def significance(region, popt):
    """
    Computes the statistical significance of the signal using the approximation:
        Z = S / sqrt(S+B),
    where S is the number of signal events and B is the estimated background within
    2 sigma of the peak.
    
    Parameters:
    region (ndarray): array of mass values within selected signal region.
    popt (ndarray): optimised parameters of signal + background model [A, mu, sigma, a, b, c].
        
    Returns:
    signal_events (int): number of signal events.
    background_events (int): number of background events.
    significance (int): statistical significance of the signal.
    """
    counts, bin_edges, bin_centres = histogram(region)  
    A, mu, sigma, a, b, c = popt
    
    peak_mask = (bin_centres > mu - 2*sigma) & (bin_centres < mu + 2*sigma)

    total_events = np.sum(counts[peak_mask])
    
    background_est = background(bin_centres, a, b, c)
    background_events = np.sum(background_est[peak_mask])
    
    signal_events = total_events - background_events
    significance = signal_events / np.sqrt(signal_events + background_events)
    
    return signal_events, background_events, significance

def chi_squared(region, popt):
    """
    This function calculates the chi squared and reduced chi squared of the data
    to show goodness-of-fit.
    
    Parameters:
    region (ndarray): array of mass values within selected signal region.
    popt (ndarray): optimised parameters of signal + background model [A, mu, sigma, a, b, c].        
    
    Returns:
    chi2 (int): chi squared value of data within signal region.
    chi2_red (int): reduced chi squared value of data within signal region.
    """
    counts, bin_edges, bin_centres = histogram(region)

    expected = signal_plus_background(bin_centres, *popt)

    chi2 = np.sum((counts - expected)**2 / np.where(expected == 0, 1, expected))
    dof = len(counts) - len(popt)

    chi2_red = chi2 / dof

    return chi2, chi2_red

def run_analysis():
    mass, region = load_data()
    
    m_peak, m_width, m_FWHM, popt = fit_signal_background(region)
    
    signal_events, background_events, sig = significance(region, popt)
    chi2, chi2_red = chi_squared(region, popt)
    
    return mass, region, m_peak, m_width, m_FWHM, popt, signal_events, background_events, sig, chi2, chi2_red
