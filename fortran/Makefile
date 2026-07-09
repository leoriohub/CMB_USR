FC = x86_64-conda-linux-gnu-gfortran
FFLAGS = -O3 -march=native -fopenmp -fopenmp-simd -ffast-math -funroll-loops -flto
LDFLAGS = -fopenmp -flto
MODULE = ms_solver_fort

all:
	LDFLAGS='$(LDFLAGS)' python -m numpy.f2py -c -m $(MODULE) ms_solver.f90 \
		--f90flags='$(FFLAGS)' \
		--opt='-O3'

clean:
	rm -f $(MODULE)*.so $(MODULE)*.cpython*.so *.mod
