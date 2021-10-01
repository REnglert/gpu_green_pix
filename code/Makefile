default: gpix

dp: sobel_dp canny_dp laplacian_dp

baseline: baseline.cu
	nvcc -o baseline.bin baseline.cu

sobel: 
	g++ -c main.cpp
	nvcc -c sobel.cu
	g++  -o gpix.bin main.o sobel.o -L/usr/local/cuda/lib64 -lcudart
	rm main.o sobel.o

canny: canny.cu
	nvcc -o canny.bin canny.cu

laplacian: laplacian.cu
	nvcc -o laplacian.bin laplacian.cu

sobel_dp: sobel_dp.cu
	nvcc -arch=compute_60 -rdc=true -o sobel_dp.bin sobel_dp.cu

canny_dp: canny_dp.cu
	nvcc -arch=compute_60 -rdc=true -o canny_dp.bin canny_dp.cu

laplacian_dp: laplacian_dp.cu
	nvcc -arch=compute_60 -rdc=true -o laplacian_dp.bin laplacian_dp.cu

clean:
	rm -f *.bin *.out
