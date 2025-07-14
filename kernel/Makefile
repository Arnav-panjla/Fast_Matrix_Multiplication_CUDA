.PHONY: make01 make02

cpu:
	nvcc -o cpu_mul cpu_mul.cu
	./cpu_mul
	
make01:
	nvcc -o trivial_mul trivial_mul.cu
	./trivial_mul

make02:
	nvcc -o mem_opt_mul mem_opt_mul.cu
	./mem_opt_mul
