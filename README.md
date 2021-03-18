To run container:
```shell script
docker run -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION gpr-sim
```

To configure EC2 instance:
```shell script
sudo yum update
sudo yum install docker git

# Install NVIDIA GPU driver
RUN wget https://us.download.nvidia.com/tesla/460.32.03/NVIDIA-Linux-x86_64-460.32.03.run
RUN sudo sh NVIDIA-Linux-x86_64-$DRIVER_VERSION.run

sudo yum install -y nvidia-docker2

sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker 
sudo systemctl enable docker.service
sudo systemctl enable containerd.service
sudo systemctl start docker.service


docker run hello-world
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi


# Add ssh file

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github

git clone git@github.com:jeanruggiero/masters-thesis.git

cd masters-thesis

# Export environment variables

aws s3 cp s3://jean-masters-thesis/geometry/geometry_spec3.csv ./
mv geometry_spec3.csv geometry_spec.csv


# Update number of cores to 8, add -gpu argument to main gpr command
docker build -t gpr-sim .
docker run -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION --gpus all gpr-sim &

```