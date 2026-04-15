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

.PHONY: all train train_diffusion train_hebrew_diffusion lib lib_diffusion lib_hebrew_diffusion clean

all: train train_diffusion train_hebrew_diffusion lib lib_diffusion lib_hebrew_diffusion

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

# Hebrew Diffusion training executable
train_hebrew_diffusion: train_hebrew_diffusion.c notorch.c notorch.h
	$(CC) $(CFLAGS) -o train_hebrew_diffusion train_hebrew_diffusion.c notorch.c -lm $(BLAS_FLAGS) 2>/dev/null || \
	$(CC) $(CFLAGS) -o train_hebrew_diffusion train_hebrew_diffusion.c notorch.c -lm
	@echo "Built: train_hebrew_diffusion"

# Shared library for HeVLM Python ctypes
lib: notorch.c notorch.h
	$(CC) $(CFLAGS) -shared -fPIC -o libnotorch.so notorch.c -lm $(BLAS_FLAGS) 2>/dev/null || \
	$(CC) $(CFLAGS) -shared -fPIC -o libnotorch.so notorch.c -lm
	@echo "Built: libnotorch.so"

# Shared library for Diffusion Python ctypes (no notorch dependency — standalone)
lib_diffusion: diffusion_engine.c
	$(CC) $(CFLAGS) -DDIFFUSION_LIB_ONLY -shared -fPIC -o libdiffusion.so diffusion_engine.c -lm
	@echo "Built: libdiffusion.so"

# Shared library for Hebrew Diffusion Python ctypes
lib_hebrew_diffusion: hebrew_diffusion_engine.c
	$(CC) $(CFLAGS) -DHEB_DIFF_LIB_ONLY -shared -fPIC -o libhebdiffusion.so hebrew_diffusion_engine.c -lm
	@echo "Built: libhebdiffusion.so"

# Standalone diffusion inference engine
diffusion_engine: diffusion_engine.c
	$(CC) $(CFLAGS) -o diffusion_engine diffusion_engine.c -lm
	@echo "Built: diffusion_engine"

# Standalone Hebrew diffusion inference engine
hebrew_diffusion_engine: hebrew_diffusion_engine.c
	$(CC) $(CFLAGS) -o hebrew_diffusion_engine hebrew_diffusion_engine.c -lm
	@echo "Built: hebrew_diffusion_engine"

clean:
	rm -f train_hevlm train_diffusion train_hebrew_diffusion diffusion_engine hebrew_diffusion_engine libnotorch.so libdiffusion.so libhebdiffusion.so *.o
