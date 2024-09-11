# ML_Deployment

# Mnist & Chest Classification Using PyTorch & Deploy on TorchServe Server Using Docker & Docker Compose
===========================================================================================

## Introduction
Mnist and Chest Classification is a deep learning project that uses PyTorch to classify Mnist and Chest X-ray images. The project also demonstrates how to deploy the model on TorchServe server using Docker and Docker Compose.

## Features
### Model Training
* Train a PyTorch model on Mnist and Chest X-ray datasets
* Implement data augmentation and preprocessing techniques
* Use transfer learning to improve model performance

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
* Clone the repository: `git clone https://github.com/your-username/mnist-chest-classification.git`
* Install dependencies: `pip install -r requirements.txt`
* Build the Docker image: `docker build -t mnist-chest-classification .`

### Running the Project
* Run the Docker container: `docker run -p 8080:8080 mnist-chest-classification`
* Access the TorchServe server: `http://localhost:8080`

## Usage
* Use the TorchServe API to send inference requests to the model
* Use the provided Python script to test the model and visualize the results

## Contributing
Contributions are welcome! If you'd like to contribute to the project, please submit a pull request or report an issue.

## License
This project is licensed under the MIT License.
