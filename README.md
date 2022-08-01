# Info
This code allows visualizing calculated energy fluences using Christian Glaser's [geoceLDF package](https://github.com/cg-laser/geoceLDF) for various types of arrays.
A custom array can be also used. Use files in CSV format where the first column is the station number and the following three are the stations x,y, and z coordinates.

# Prerequisites:

- geoceLDF module converted to Python3:
git clone https://github.com/F-Tomas/geoceLDF.git

- Radio tools:
git clone https://github.com/nu-radio/radiotools.git

Do not forget to add them to your python environmental variable `$PYTHONPATH`!
E.g.:

    export PYTHONPATH=$PYTHONPATH"/home/tomas/github/work:/home/tomas/github/work/geoceLDF:/home/tomas/github/work/radiotools:"

- other prerequisites can be installed standardly via python -m pip install other_prerequisity

# 
The average Xmax values in file "Xmax.dat" used in the calculation of the energy fluence were taken from ref.[1] up to lg(ECR) = 19.55 and from ref.[2] up to lg(ECR) = 19.85 and, for the highest energies extrapolated from [2]. Values were directly taken from Pierre Auger Collaboration's Internal publication GAP 2020-008 authored by A. Saftoiu and T. Huege.<br>
[1] A. AAb et al. Pierre Auger Collaboration, Phys. Rev. D 90 (2014) 122005; arXiv:1409.4809v3.<br>
[2] P. Sanchez-Lucas et al. Pierre Auger Collaboration, ICRC2017 Busan, Korea; arXiv:1708.06592v2.<br>

