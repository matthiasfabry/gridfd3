
all : clean gridfd3

clean :
	rm -f ./bin/* ./src/*.o

gridfd3 : src/fd3grid.o src/fd3sep.o src/triorb.o src/kepler.o src/mxfuns.o
	${CC} -Wall src/fd3grid.o src/fd3sep.o src/triorb.o src/kepler.o src/mxfuns.o \
	-lgsl -lgslcblas -lm -o bin/$@
