all:
	stack ghc Main.hs
	make clean

clean:
	rm ./NeuralNet/*.o ./NeuralNet/*.hi
	rm *.o *.hi
