nvcc = /usr/local/cuda-8.0/bin/nvcc
cudalib = /usr/local/cuda-8.0/lib64/
#tensorflow = /usr/local/lib/python2.7/dist-packages/tensorflow/include
tensorflow = /home/aadc/.virtualenvs/PointTAM/lib/python3.5/site-packages/tensorflow/include

all: depthestimate/tf_nndistance_so.so depthestimate/render_balls_so.so
.PHONY : all

depthestimate/tf_nndistance_so.so: depthestimate/tf_nndistance_g.cu.o depthestimate/tf_nndistance.cpp
	g++ -std=c++11 depthestimate/tf_nndistance.cpp depthestimate/tf_nndistance_g.cu.o -o depthestimate/tf_nndistance_so.so -shared -fPIC -I $(tensorflow) -I $(tensorflow)/external/nsync/public -lcudart -L $(cudalib) -L $(tensorflow)/.. -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

depthestimate/tf_nndistance_g.cu.o: depthestimate/tf_nndistance_g.cu
	$(nvcc) -gencode=arch=compute_50,code=sm_50 -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -c -o depthestimate/tf_nndistance_g.cu.o depthestimate/tf_nndistance_g.cu -I $(tensorflow) -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -O2

depthestimate/render_balls_so.so: depthestimate/render_balls_so.cpp
	g++ -std=c++11 depthestimate/render_balls_so.cpp -o depthestimate/render_balls_so.so -shared -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=0

clean:
	rm depthestimate/render_balls_so.so depthestimate/tf_nndistance_g.cu.o depthestimate/tf_nndistance_so.so
