Acknowledging
=============

If you use ``21cmFAST v3+`` in your research please cite both of:

    Murray et al., (2020). 21cmFAST v3: A Python-integrated C code for generating 3D
    realizations of the cosmic 21cm signal. Journal of Open Source Software, 5(54),
    2582, https://doi.org/10.21105/joss.02582

    Andrei Mesinger, Steven Furlanetto and Renyue Cen, "21CMFAST: a fast, seminumerical
    simulation of the high-redshift 21-cm signal", Monthly Notices of the Royal
    Astronomical Society, Volume 411, Issue 2, pp. 955-972 (2011),
    https://ui.adsabs.harvard.edu/link_gateway/2011MNRAS.411..955M/doi:10.1111/j.1365-2966.2010.17731.x

In addition, the following papers introduce various features into ``21cmFAST``. If you use
these features, please cite the relevant papers.

Lyman alpha multiple scattering

    Flitter, J., Munoz, J. B., Mesinger, A.,
    "Semi-analytical approach to Lyman-alpha multiple-scattering in 21-cm signal simulations",
    eprint arXiv:2601.14360, 2026. https://doi.org/10.48550/arXiv.2601.14360.

Discrete Halo Sampler / version 4:

    Davies, J. E., Mesinger, A., Murray, S. G.,
    "Efficient simulation of discrete galaxy populations and associated radiation fields during the first billion years",
    Astronomy and Astrophysics, vol. 701, A. 236, 2025. https://doi.org/10.1051/0004-6361/202554951.

Photon non-conservation correction:

    Park, J., Greig, B., Mesinger, A.,
    "Calibrating excursion set reionization models to approximately conserve ionizing photons",
    Monthly Notices of the Royal Astronomical Society, vol. 517, no. 1,
    pp 192-200, 2022 https://doi.org/10.1093/mnras/stac2756.

Mini-halos:

    Muñoz, J.B., Qin, Y., Mesinger, A., Murray, S., Greig, B., and Mason, C.,
    "The Impact of the First Galaxies on Cosmic Dawn and Reionization",
    Monthly Notices of the Royal Astronomical Society, vol. 511, no. 3,
    pp 3657-3681, 2022 https://doi.org/10.1093/mnras/stac185
    (for DM-baryon relative velocities)

    Qin, Y., Mesinger, A., Park, J., Greig, B., and Muñoz, J. B.,
    “A tale of two sites - I. Inferring the properties of minihalo-hosted galaxies from
    current observations”, Monthly Notices of the Royal Astronomical Society, vol. 495,
    no. 1, pp. 123–140, 2020. https://doi.org/10.1093/mnras/staa1131.
    (for Lyman-Werner and first implementation)

Mass-dependent ionizing efficiency:

    Park, J., Mesinger, A., Greig, B., and Gillet, N.,
    “Inferring the astrophysics of reionization and cosmic dawn from galaxy luminosity
    functions and the 21-cm signal”, Monthly Notices of the Royal Astronomical Society,
    vol. 484, no. 1, pp. 933–949, 2019. https://doi.org/10.1093/mnras/stz032.

Lightcone and redshift space distortions:

    Greig, B., Mesinger, A.,
    "21CMMC with a 3D light-cone: the impact of the co-evolution approximation on the astrophysics of reionisation and cosmic dawn.",
    Monthly Notices of the Royal Astronomical Society, vol. 477, no. 3,
    pp 3217-3229, 2018. https://doi.org/10.1093/mnras/sty796.

Inhomogeneous recombination:

    Sobacchi, E., Mesinger, A.,
    "Inhomogeneous recombinations during cosmic reionization",
    Monthly Notices of the Royal Astronomical Society, vol. 440, no. 2,
    pp 1662-1673, 2014. https://doi.org/10.1093/mnras/stu377.

If you are unsure which modules are used within your simulations, we provide a handy function
to print out which works to refer ``py21cmfast.utils.show_references``, which accepts a single instance of
the ``InputParameters`` class and shows which papers are relevant for your simulation.
