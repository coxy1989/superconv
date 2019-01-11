- config nvidia docker: https://gist.github.com/Brainiarc7/a8ab5f89494d053003454efc3be2d2ef 
- verify nvidia docker: sudo docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
- docker run -it -v $PWD:/mount --runtime=nvidia --shm-size=8g --rm caffe
- download cifar10 and verify all well by running cifar10 tutorial: https://github.com/BVLC/caffe/tree/master/examples/cifar10
- download mnist and verify all well by running mnist tutorial: http://caffe.berkeleyvision.org/gathered/examples/mnist.html


