Install of python3 including numpy, scipy, and matplotlib is expected.  Pycircstat library is used, code needs to be built for your machine.  See their documentation at https://github.com/circstat/pycircstat.  

Special files:
`model.py ` — contains all inference machinery
`myutils.py` — contains utils and params used in simulations

All other files are simulations.  When you run a simulation (e.g., `python cuevariance.py`), figures will be output into the figures folder.  An easy way to run all of the simulations (except for the simulations included in `partitionParams.py`) is to run `runall.py`.  
If you want to run all the simulations for the paper, 
```
python runall.py
python partitionParams.py
```
