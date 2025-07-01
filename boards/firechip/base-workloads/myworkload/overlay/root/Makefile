# Makefile for MLP benchmark with M5 library

# riscv64-unknown-linux-gnu-gcc -Ofast -march=rv64gc -flto -std=c11 -o matrix matrix.c

# Compiler settings
CC = riscv64-unknown-linux-gnu-g++
# CC = riscv64-unknown-linux-gnu-gcc

CFLAGS = -O2 -Wall -g -static -DRISCV -I./ -I./asm/generic

# Current directory path (where libm5.a is located)
M5_PATH = $(shell pwd)

# Linker flags - add the missing flags for proper linking
LDFLAGS = -L$(M5_PATH) -lm5 -Wl,--no-as-needed -lrt -lpthread -lm

# Target executable
TARGET = bankpll


$(TARGET): bank_pll.cc
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(TARGET)