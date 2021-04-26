all: main
			./main

main: main.o
			g++ -I./ -o main main.o

main.o: main.cpp
				g++ -I./ -c main.cpp

clean: 
				rm main.o main