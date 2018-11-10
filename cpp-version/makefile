CXXSTD := c++17


default: neuronet

%.o: %.cpp
	g++ -c $^ -std=$(CXXSTD)

neuronet: main.o utils.o
	g++ -o $@ $^ -std=$(CXXSTD)
