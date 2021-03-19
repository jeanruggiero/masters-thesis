sudo yum install -y git gcc

# Driver installation prerequisites
sudo yum install -y kernel-devel-$(uname -r) kernel-headers-$(uname -r)


# Install NVIDIA GPU driver
wget https://us.download.nvidia.com/tesla/460.32.03/NVIDIA-Linux-x86_64-460.32.03.run
sudo sh NVIDIA-Linux-x86_64-460.32.03.run


#sudo usermod -aG docker $USER
#newgrp docker
#sudo systemctl enable docker.service
#sudo systemctl enable containerd.service
#sudo systemctl start docker.service

#
#docker run hello-world
#docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi


# Export environment variables

aws s3 cp s3://jean-masters-thesis/geometry/geometry_spec3.csv ~/masters-thesis
mv ~/geometry_spec3.csv ~/geometry_spec.csv