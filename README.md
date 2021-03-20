To run container:
```shell script
docker run -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION gpr-sim
```

To configure EC2 instance:
Select Deep Learning Base AMI (Amazon Linux 2) machine image
p3.2xlarge instance type
```shell script
sudo yum update -y
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-11.0 /usr/local/cuda
```


```shell script
sudo yum install -y docker git gcc

# Driver installation prerequisites
sudo yum install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)


# Install NVIDIA GPU driver
wget https://us.download.nvidia.com/tesla/460.32.03/NVIDIA-Linux-x86_64-460.32.03.run
sudo sh NVIDIA-Linux-x86_64-460.32.03.run

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo
sudo yum clean expire-cache
sudo yum install nvidia-container-toolkit -y

sudo usermod -aG docker $USER
newgrp docker 
sudo systemctl enable docker.service
sudo systemctl enable containerd.service
sudo systemctl start docker.service


docker run hello-world
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi


# Add ssh file
vim ~/.ssh/git_thesis
chmod 400 ~/.ssh/git_thesis 

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/git_thesis

git clone git@github.com:jeanruggiero/masters-thesis.git

cd masters-thesis

# Export environment variables

aws s3 cp s3://jean-masters-thesis/geometry/geometry_spec3.csv ./
mv geometry_spec3.csv geometry_spec.csv


sudo yum install nvidia-container-runtime




# Update number of cores to 8, add -gpu argument to main gpr command
docker build -t gpr-sim .
docker run -e AWS_ACCESS_KEY_ID -e AWS_SECRET_ACCESS_KEY -e AWS_DEFAULT_REGION --gpus all gpr-sim &

```