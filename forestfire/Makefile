all:	a.out
CC  = 	g++
LIB = 	-lglut -lGLEW -lGLU -lGL -lOpenCL

cpp.o:
	$(CC) -c $<

a.out: ../common/SakiyaMas.o Source.o
	$(CC) -o $@ Source.o ../common/SakiyaMas.o $(LIB)

clean:
	@rm -rf a.out *.o
