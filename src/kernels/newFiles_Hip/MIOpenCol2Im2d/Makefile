# Makefile for HIP project

# HIP path and compiler
HIP_PATH ?= /opt/rocm/hip
HIPCC = $(HIP_PATH)/bin/hipcc

# Source files
SRC = Main_MIOpenCol2Im2d.cpp
OBJ = $(SRC:.cpp=.o)

# Include and library paths
INCLUDES = -I$(HIP_PATH)/include
LIBS = -L$(HIP_PATH)/lib

# Compiler flags
CXXFLAGS = -Wall -g $(INCLUDES)

# Target
TARGET = MIOpenCol2Im2d

# Rules
all: $(TARGET)

$(TARGET): $(OBJ)
	$(HIPCC) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(HIPCC) -c $(CXXFLAGS) $< -o $@

# Clean
clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
