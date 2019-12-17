
all: fd2grid

clean :
	rm fd2grid *.o

fd2grid : fd2grid.o fd2sep.o orb.o kepler.o mxfuns.o
	${CC} -Wall fd2grid.o fd2sep.o orb.o kepler.o mxfuns.o \
		-lgsl -lgslcblas -lm -o $@

.c.o :
	${CC} -Wall -c $<
