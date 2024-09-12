![image](https://github.com/user-attachments/assets/a13b48e5-4c5f-40f8-baa4-b7a6fd59b3dc)

<h1 style="text-align: center;">ML_Deployment</h1>

# Mnist & Chest Xray-Image Classification Using PyTorch & Deploy on TorchServe Server Using Docker & Docker Compose
============================================================================================

## Introduction
Mnist and Chest Classification is a deep learning project that uses PyTorch to classify Mnist and Chest X-ray images. The project also demonstrates how to deploy the model on TorchServe server using Docker and Docker Compose.

## Technologies
* <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=gray" width="100" height="30" alt="PyTorch Badge">
* <img src="https://img.shields.io/badge/TorchServe-FF69B4?style=for-the-badge&logo =pytorch&logoColor=white" width="100" height="30" alt="TorchServe Badge">
* <img src="https://user-images.githubusercontent.com/25181517/192107854-765620d7-f909-4953-a6da-36e1ef69eea6.png" width="50" height="50" alt="Icon">
* <img src="https://user-images.githubusercontent.com/25181517/192108374-8da61ba1-99ec-41d7-80b8-fb2f7c0a4948.png" width="50" height="50" alt="Icon">
* <img src="https://user-images.githubusercontent.com/25181517/192108372-f71d70ac-7ae6-4c0d-8395-51d8870c2ef0.png" width="50" height="50" alt="Icon">
* <img src="https://user-images.githubusercontent.com/25181517/183914128-3fc88b4a-4ac1-40e6-9443-9a30182379b7.png" width="50" height="50" alt="Icon">
* <img src="https://user-images.githubusercontent.com/25181517/192108891-d86b6220-e232-423a-bf5f-90903e6887c3.png" width="50" height="50" alt="Icon">
* <img src="https://user-images.githubusercontent.com/25181517/183898054-b3d693d4-dafb-4808-a509-bab54cf5de34.png" width="50" height="50" alt="Icon">
* <img src="https://user-images.githubusercontent.com/25181517/183898674-75a4a1b1-f960-4ea9-abcb-637170a00a75.png" width="50" height="50" alt="Icon">
* <img src="https://user-images.githubusercontent.com/25181517/192158954-f88b5814-d510-4564-b285-dff7d6400dad.png" width="50" height="50" alt="Icon">
* <img src="https://github.com/marwin1991/profile-technology-icons/assets/76012086/24b02d77-2f28-43c7-b5d6-e15e3395851b" width="50" height="50" alt="Icon">
* <img src="https://github.com/marwin1991/profile-technology-icons/assets/76012086/4ec200c2-acdf-4c42-b419-cd49cba3d09f" width="50" height="50" alt="Icon">
* <img src="https://user-images.githubusercontent.com/25181517/183423775-2276e25d-d43d-4e58-890b-edbc88e915f7.png" width="50" height="50" alt="Icon">
* <img src="https://user-images.githubusercontent.com/25181517/183423507-c056a6f9-1ba8-4312-a350-19bcbc5a8697.png" width="50" height="50" alt="Icon">
* <img src="https://user-images.githubusercontent.com/25181517/192158606-7c2ef6bd-6e04-47cf-b5bc-da2797cb5bda.png" width="50" height="50" alt="Icon">
* <img src="https://user-images.githubusercontent.com/25181517/117207330-263ba280-adf4-11eb-9b97-0ac5b40bc3be.png" width="50" height="50" alt="Icon">
* <img src="https://github.com/marwin1991/profile-technology-icons/assets/76662862/2481dc48-be6b-4ebb-9e8c-3b957efe69fa" width="50" height="50" alt="Icon">
* <img src="https://img.shields.io/badge/Shell_Script-121011?style=for-the-badge&logo=gnu-bash&logoColor=white" width="100" height="30" alt="Shell Script Badge">




## Features
### Model Training
* Train a PyTorch model on Mnist and Chest X-ray datasets
* Implement data augmentation and preprocessing techniques
* implement model from scratch and save on .pt files

### Model Deployment
* Deploy the trained model on TorchServe server using Docker
* Use Docker Compose to manage multiple containers
* Implement model serving and inference using TorchServe API

### Docker and Docker Compose
* Create a Docker image for the PyTorch model
* Use Docker Compose to manage multiple containers for the model, TorchServe server, and database

## Getting Started
### Prerequisites
* Python 3.8 or later
* PyTorch 1.9 or later
* Docker and Docker Compose installed on your system

### Installation
* Clone the repository: `git clone https://github.com/mobinghasemi/ML_Deployment.git`

### Running the Project
#### MnistApp
* copy mnist_index.html to the /app/templates/index.html
* copy mnist_app.py to the /app/app.py
##### VScode Terminal
```dockerfile
docker compose up --bulid
```
##### PC Terminal
```dockerfile
docker ps
docker exec -u 0 -it container-id(torchserve container) /bin/bash
```
```bash
cd /
cd /codes
ls -ltrha
torch-model-archiver --model-name mnist --version 2.0 --model-file arch.py --serialized-file mnist.pt --handler mnist_handler_base.py --force
cp mnist.mar /home/model-server/model-store
torchserve --stop
torchserve --start --model-store /home/model-server/model-store --models mnist=mnist.mar --disable-token-auth --enable-model-api --ts-config /home/model-server/config.properties    
```
##### Browser
```curl
localhost:9696
```

#### XrayApp
* copy xray_index.html to the /app/templates/index.html
* copy xray_app.py to the /app/app.py
##### VScode Terminal
```dockerfile
docker compose up --bulid
```
##### PC Terminal
```dockerfile
docker ps
docker exec -u 0 -it container-id(torchserve container) /bin/bash
```
```bash
cd /
cd /codes
ls -ltrha
torch-model-archiver --model-name xray --version 1.0 --model-file xray_arch.py --serialized-file xray.pt --handler xray_handler_base.py --force
cp xray.mar /home/model-server/model-store
torchserve --stop
torchserve --start --model-store /home/model-server/model-store --models xray=xray.mar --disable-token-auth --enable-model-api --ts-config /home/model-server/config.properties    
```
##### Browser
```curl
localhost:9696
```


## Outputs
### MnistApp
![MnistApp](output_images/mnist1.png)
![MnistApp](output_images/mnist2.png)
### XrayApp
![XrayApp](output_images/xray1.png)
![XrayApp](output_images/xray2.png)


## Usage
* Use the TorchServe API to send inference requests to the model
* Use the provided Python script to test the model and visualize the results

## Contributing
Contributions are welcome! If you'd like to contribute to the project, please submit a pull request or report an issue.

## License
This project is licensed under the MIT License.
