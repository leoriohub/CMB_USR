FC = x86_64-conda-linux-gnu-gfortran
# Must be the same toolchain family as FC: -flto emits version-specific LTO
# bytecode, so linking with a different GCC than the one that compiled the
# objects fails at link time ("generated with LTO version 14.0 instead of 16.0").
CC = x86_64-conda-linux-gnu-gcc
FFLAGS = -O3 -march=native -fopenmp -fopenmp-simd -ffast-math -funroll-loops -flto
LDFLAGS = -fopenmp -flto
MODULE = ms_solver_fort

all:
	FC='$(FC)' CC='$(CC)' LDFLAGS='$(LDFLAGS)' python -m numpy.f2py -c -m $(MODULE) ms_solver.f90 \
		--f90flags='$(FFLAGS)' \
		--opt='-O3'

clean:
	rm -f $(MODULE)*.so $(MODULE)*.cpython*.so *.mod
