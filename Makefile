default:
	c++ -O3 -Wall -shared -std=c++11 -fPIC $(python -m pybind11 --includes) src/simple_ml_ext.cpp -o src/simple_ml_ext.so