# Makefile for disparity_determ_top.c with M5 library

# Compiler settings
CC = riscv64-unknown-linux-gnu-gcc
CFLAGS = -I./ -I./asm/generic

# Current directory path (where libm5.a is located)
M5_PATH = $(shell pwd)

# Linker flags
LDFLAGS = -L$(M5_PATH) -lm5

# Target executable
TARGET = disparity_determ_top

all: $(TARGET)

$(TARGET): disparity_determ_top.c
	$(CC) -o $@ $< $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(TARGET)