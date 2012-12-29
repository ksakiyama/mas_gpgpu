all:	a.out
CC  = 	gcc
LIB = 	-lglut -lGLEW -lGLU -lGL -lOpenCL

.cpp.o:
	$(CC) -c $<

a.out: ../../common/SakiyaMas.o Source.o
	$(CC) -o $@ Source.o ../../common/SakiyaMas.o $(LIB) -O3

clean:
	@rm -rf a.out *.o
