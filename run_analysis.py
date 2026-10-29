from dimuon_analysis import (load_data, fit_signal_background, plot_data, 
    plot_fit, significance, chi_squared)

def main():
    mass, region = load_data()

    m_peak, m_width, m_FWHM, popt = fit_signal_background(region)

    signal_events, background_events, sig = significance(region, popt)
    chi2, chi2_red = chi_squared(region, popt)
    
    choice = str(input("Choose a graph: Raw data (R), signal region (S), or both? ")).upper()
    if choice == "R":
        plot_data(mass)
    elif choice == "S":
        plot_fit(region, popt)
    else:
        plot_data(mass)
        plot_fit(region, popt)
    

    print("\n ----- Fit Results ----")
    print("Estimated particle mass:", m_peak, "GeV")
    print("Gaussian width (σ):", m_width, "GeV")
    print("Gaussian FWHM:", m_FWHM, "GeV")

    print("\n ----- Signal Statistics ----")
    print(f"Signal events: {signal_events:.4f}")
    print(f"Background events: {background_events:.4f}")
    print(f"Statistical significance: {sig:.4f} σ")

    print("\n ----- Goodness of Fit ----")
    print(f"Chi-squared: {chi2:.4f}")
    print(f"Reduced chi-squared: {chi2_red:.4f}")
    
if __name__ == "__main__":
    main()