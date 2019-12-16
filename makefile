
all: fd3grid

clean :
	rm fd3grid *.o

fd3grid : fd3grid.o fd3sep.o orb.o kepler.o mxfuns.o
	${CC} -Wall fd3grid.o fd3sep.o orb.o kepler.o mxfuns.o \
		-lgsl -lgslcblas -lm -o $@

.c.o :
	${CC} -Wall -c $<
