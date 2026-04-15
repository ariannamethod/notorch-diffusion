# ariannamethod/ — notorch build
# Builds training executables and shared libraries for Python/ctypes

CC = cc
CFLAGS = -O2 -Wall -Wextra -std=c11 -I.

UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
  BLAS_FLAGS = -DUSE_BLAS -DACCELERATE -DACCELERATE_NEW_LAPACK -framework Accelerate
endif
ifeq ($(UNAME), Linux)
  BLAS_FLAGS = -DUSE_BLAS -lopenblas
endif

.PHONY: all train train_diffusion lib lib_diffusion clean

all: train train_diffusion lib lib_diffusion

# HeVLM training executable
train: train_hevlm.c notorch.c notorch.h
	$(CC) $(CFLAGS) -o train_hevlm train_hevlm.c notorch.c -lm $(BLAS_FLAGS) 2>/dev/null || \
	$(CC) $(CFLAGS) -o train_hevlm train_hevlm.c notorch.c -lm
	@echo "Built: train_hevlm"

# Diffusion training executable
train_diffusion: train_diffusion.c notorch.c notorch.h
	$(CC) $(CFLAGS) -o train_diffusion train_diffusion.c notorch.c -lm $(BLAS_FLAGS) 2>/dev/null || \
	$(CC) $(CFLAGS) -o train_diffusion train_diffusion.c notorch.c -lm
	@echo "Built: train_diffusion"

# Shared library for HeVLM Python ctypes
lib: notorch.c notorch.h
	$(CC) $(CFLAGS) -shared -fPIC -o libnotorch.so notorch.c -lm $(BLAS_FLAGS) 2>/dev/null || \
	$(CC) $(CFLAGS) -shared -fPIC -o libnotorch.so notorch.c -lm
	@echo "Built: libnotorch.so"

# Shared library for Diffusion Python ctypes (no notorch dependency — standalone)
lib_diffusion: diffusion_engine.c
	$(CC) $(CFLAGS) -DDIFFUSION_LIB_ONLY -shared -fPIC -o libdiffusion.so diffusion_engine.c -lm
	@echo "Built: libdiffusion.so"

# Standalone diffusion inference engine
diffusion_engine: diffusion_engine.c
	$(CC) $(CFLAGS) -o diffusion_engine diffusion_engine.c -lm
	@echo "Built: diffusion_engine"

clean:
	rm -f train_hevlm train_diffusion diffusion_engine libnotorch.so libdiffusion.so *.o
