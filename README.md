```
    ^ ^
  <(O_o)>
   ( . )
----"-"------->>
```

# Quaia redshifts and the Milky Way

Code for the paper "Quaia quasar redshifts depend on direction through the Milky Way" (J. Fuentes Aguilar, 2026). Every number and figure in the paper comes from these scripts.

Archived on Zenodo, https://doi.org/10.5281/zenodo.24144793

## Data

The catalogues are public and are not included here. Put them in a folder `data/` in this repository:

```
data/
  quaia_G20.0.fits
  quaia_G20.5.fits
  selection_function_NSIDE64_G20.0.fits
  selection_function_NSIDE64_G20.5.fits
  templates/selection_function_template_maps/map_*_NSIDE64.npy
  desi/QSO_NGC_clustering.dat.fits
  desi/QSO_SGC_clustering.dat.fits
```

The Quaia catalogues, selection functions and template maps are from the Quaia release of Storey-Fisher et al. (2024), version 1.0.0, at https://doi.org/10.5281/zenodo.10403370. Unzip `selection_function_template_maps.zip` into `data/templates/`. The DESI files are the DR1 quasar clustering catalogues, version 1.5, at https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/

## Running

Run everything from the top of the repository, after creating `out/`:

```
mkdir out
python arf_theory.py
python mocks.py
python results.py
python fnl.py
python fnl.py curve
python fnl.py correction
python fnl.py heldout
python arf_fnl.py
python figures.py
```

The order matters. `mocks.py` needs the theory spectra from `arf_theory.py`, `results.py` needs those and the mocks, `fnl.py curve`, `correction` and `heldout` need a plain `fnl.py` run first, and `figures.py` reads the results of all of them. The numbers are printed to the terminal. Redirect them to a file if you want to keep them. `arf_fnl.py` takes about ten minutes. `mocks.py` makes 200 mock skies by default, and `python mocks.py 50` makes fewer.

What each script gives, with the sections of the paper:

- `results.py` the dependence of the redshifts on the templates, the comparison with DESI, the maps of the mean redshift and the dipole (sections 3 to 5.2). `python results.py 4.1` runs one part only, with the labels 3, 4.1, 4.2 and 4.3 for the errors, the templates and DESI, the maps and the dipole.
- `mocks.py` the three kinds of mock skies (section 3.3).
- `arf_theory.py` linear theory for the maps of the mean redshift (section 3.2).
- `fnl.py` bin migration and the bias of f_NL from the counts (section 5.3).
- `arf_fnl.py` the bias of f_NL from the maps of the mean redshift (section 5.3).
- `figures.py` figures 1 to 4, written to `figs/`.
- `quaia.py`, `theory.py` and `plotstyle.py` are shared by the others.

## Python

Tested with Python 3.13 and numpy 2.5.4, scipy 1.18.1, astropy 8.0.2, healpy 1.20.1, camb 2.0.5 and matplotlib 3.11.2.

## Licence

MIT, see `LICENSE`. If you use the code, please cite the paper.
