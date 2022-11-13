# we have noisy measurements
# we want origin, integer coordinates, and basis of lattice

param ndim >= 1; # number of dimensions of the problem, usually 2, generalizes to however many
param nmes >= 1; # number of noisy measurements       
param meas{1..nmes, 1..ndim}; # noisy measurements  

var or{1..ndim}; # origin of lattice
var bv{1..ndim, 1..ndim}; # basis of lattice
var coord{1..nmes, 1..ndim} integer; # lattice coordinates of points

var errs{1..nmes, 1..ndim}; # auxiliary variable for errors of each measurement in each dimension

minimize obj: 
	sum{i in 1..ndim, j in 1..nmes} errs[j, i] * errs[j, i];

# we define the errors
subject to DefErrs {i in 1..ndim, j in 1..nmes}:
	errs[j, i] = -meas[j, i] + or[i] + (sum {k in 1..ndim} coord[j, i] * bv[k, i]);
	
# we impose that the coordinates must be between -n and n
subject to CoordRest1 {i in 1..ndim, j in 1..nmes}:
	coord[j, i] <= nmes;
subject to CoordRest2 {i in 1..ndim, j in 1..nmes}:
	coord[j, i] >= -nmes;
