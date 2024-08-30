torch-model-archiver --model-name mnist --version 2.0 --model-file arch.py --serialized-file mnist.pt --handler mnist_handler_base.py --force

mv mnist.mar model-store/

docker run --rm -it -p 8080:8080 -p 8081:8081 --name mar -v $(pwd)/model-store:/home/model-server/model-store -v $(pwd):/home/model-server/examples pytorch/torchserve:latest torchserve --start --model-store model-store --models mnist=mnist.mar

curl http://127.0.0.1:8081/models/

curl http://127.0.0.1:8080/predictions/mnist -T 3.png
