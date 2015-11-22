CC=gcc

prof:
	gcc -o prof -O0 profiler.c -L./papi/lib -I./papi/include -lpapi

clean:
	rm -f *.o prof

run:
	./prof
