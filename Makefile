all:
	g++ -c -o layer_norm.o layer_norm.cpp
	g++ -shared -o libln.so -fPIC layer_norm.o
	nvcc -o layer_norm.out layer_norm.cu layer_norm.o
	g++ -I eigen-3.4.0 -o layer_norm_eigen.out layer_norm_eigen.cpp layer_norm.o -O2 -mavx2 -DNDEBUG

clean:
	rm -rf *.so *.o *.out
