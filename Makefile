all:
	g++ -c -o layer_norm.o layer_norm.cpp
	g++ -shared -o libln.so -fPIC layer_norm.o

clean:
	rm -rf *.so *.o
