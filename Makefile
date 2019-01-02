main:
	g++ -o 2 2.cpp `pkg-config --cflags --libs opencv`
	
run:
	./2

clean:
	rm 2
	
