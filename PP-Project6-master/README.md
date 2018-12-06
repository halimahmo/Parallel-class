# PP-Project6 ![](https://github.com/bab178/PP-Project6/blob/master/fractal.gif)
![ScreenShot](https://github.com/bab178/PP-Project6/blob/master/Screenshot_1.png)

Initial modules and libraries to load on TXState Lab Machines:
---------------
```
module add openmpi-x86_64
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
```

Compiling on TXState Lab Machines:
----------------
```
mpicxx -march=native -O3 -fopenmp -c fractal_hyb1.cpp -o Cfractal.o
nvcc -O3 -arch=sm_20 -c fractal_hyb.cu -o CUfractal.o
mpicxx -march=native -O3 -fopenmp Cfractal.o CUfractal.o -lcudart -L /usr/local/cuda/lib64/ -o fractal_hyb
```

Create a sample .gif of the fractal after running program:
---------------------
```
mpirun -n 2 ./fractal_hyb 32 32 32
convert -delay 1x30 fractal1*.bmp fractal.gif
rm fractal1*.bmp
```

Initial modules on Stampede:
----------------
```
module load cuda
```

Compiling hybrid parallel programs on Stampede:
------------------------
```
mpicc -xhost -openmp -O3 -c fractal_hyb.cpp -o Cfractal.o
nvcc -O3 -arch=sm_35 -c fractal_hyb.cu -o CUfractal.o
mpicc -xhost -openmp -O3 Cfractal.o CUfractal.o -lcudart -L$TACC_CUDA_LIB -o fractal_hyb
```
