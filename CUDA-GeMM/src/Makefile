NVCC = nvcc
NVCC_FLAGS = -O3 -use_fast_math -arch=sm_75

SRCS = main.cu kernel.cu
OBJS = $(SRCS:.cu=.o)
TARGET = gemm

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)