CXX_FLAGS = -std=c++11 -O3
LIBRARY = -L$(AMDAPPSDKROOT)/lib/x86_64 -lOpenCL

TARGET = triad.out sort.out

all: $(TARGET)

.SUFFIXES:
.SUFFIXES: .cpp .out
.cpp.out:
	g++ $(CXX_FLAGS) $< $(LIBRARY) -o $@

clean:
	rm -f *~ $(TARGET) core.*
