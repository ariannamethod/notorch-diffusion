# ariannamethod/ — notorch build
# Builds training executable and shared library for Python/ctypes

CC = cc
CFLAGS = -O2 -Wall -Wextra -std=c11 -I.

UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
  BLAS_FLAGS = -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate
endif
ifeq ($(UNAME), Linux)
  BLAS_FLAGS = -DUSE_BLAS -lopenblas
endif

.PHONY: all train lib clean

all: train lib

# Training executable
train: train_hevlm.c notorch.c notorch.h
	$(CC) $(CFLAGS) -o train_hevlm train_hevlm.c notorch.c -lm $(BLAS_FLAGS) 2>/dev/null || \
	$(CC) $(CFLAGS) -o train_hevlm train_hevlm.c notorch.c -lm
	@echo "Built: train_hevlm"

# Shared library for Python ctypes
lib: notorch.c notorch.h
	$(CC) $(CFLAGS) -shared -fPIC -o libnotorch.so notorch.c -lm $(BLAS_FLAGS) 2>/dev/null || \
	$(CC) $(CFLAGS) -shared -fPIC -o libnotorch.so notorch.c -lm
	@echo "Built: libnotorch.so"

clean:
	rm -f train_hevlm libnotorch.so *.o
