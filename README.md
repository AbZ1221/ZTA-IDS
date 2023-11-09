# Simulate Normal/Attack Requests
To Simulate the requests through web, we need to implement a website on a server and sniff all incoming requests. On the other hand we need many other clients (on different servers) that send the requests to the server using different ports and methods.

The simplest way to implement many Clients and some Servers with different IPs and also the range of IPs, is by Virtual Machine. We use Docker to implement Servers and Clients. We use docker-compose with a dedicated docker network with our interested range of IP.

## Server Docker
Since a module of the server needs interaction we need to pull source docker image and install the required modules ourself. First we should pull the docker source image ubuntu:20.04 by `docker pull ubuntu:20.04`, then we need to create a docker container by `docker run --rm -it --name ubuntu ubuntu:20.04 bash`. Now, we are in docker container and we can install the modules using these commands:
```bash
apt update

apt install build-essential git openssh-server -y
apt update && apt install -yqq daemonize dbus-user-session fontconfig iproute2 curl
daemonize /usr/bin/unshare --fork --pid --mount-proc /lib/systemd/systemd --system-unit=basic.target

/etc/init.d/ssh start

curl -fsSL https://deb.nodesource.com/setup_16.x | bash -
apt install yarn nodejs vim tcpdump -y

wget https://go.dev/dl/go1.20.4.linux-amd64.tar.gz
tar -C /usr/local -xzf go*.linux-amd64.tar.gz

vim /root/.bashrc
echo "export PATH=$PATH:/usr/local/go/bin" >> /root/.bashrc

source /root/.bashrc

git clone https://github.com/go-gitea/gitea.git
cd /gitea
TAGS="bindata sqlite sqlite_unlock_notify" make build
chmod 777 -R /gitea
useradd -ms /bin/bash git
```
, open another terminal in host and copy ssh config file from host to docker container by `docker cp sshd_config ubuntu:/` and inside the docker container move it to the ssh config path using `mv /sshd_config /etc/ssh/`.

Open another terminal in host, and use this command to save the changes:
```bash
docker commit ubuntu gitea
```
Now, we have a new docker image named gitea:latest which our required modules are installed in.

## Client Docker
We use kalilinux as client docker because it's light and we can run most attack simulations on it. It does not need any interaction with command line, so we can use Docker image file to build it automatically.

Run this code to build client docker image.
```bash
cd Client
docker build -t kalilinux/kali-rolling:latest .
cd ..
```

*Please not that in "dockerd-entrypoint.sh" file, there exists an IP address which is for server. This IP should be changed according to the used and taken IP address of the server docker container.*

## Run server and docker network
We use docker-compose to run server and build the network of desired IP ranges. Edit docker-compose and change subnet to our desired IP range, then make it up using:

```bash
docker-compose up
```

Now we can open the gitea web page in browser with address of our host.

### Run ssh servise in server docker
We should run ssh-server service in server by hand. Let's get into the docker container by `docker exec -it gitea bash` and run ssh service using `/etc/init.d/ssh restart`. Finally, run tcpdump based on the network device and desired filename, `tcpdump -i netdevice -w filename.pcap`.

## Run Clients
*The client docker image is built by the IP address of server after docker-compose up command*

We can run this command as many times as we want to have different client IP address in the range of server IP address:
```bash
docker run --rm -it -u root --network root_gitea --name client1 client:latest
```

## Install and Run firewall on host
Install and active firewall to protect the host being attacked on other ports than are desired and under evaluation.
Install firewall on ubuntu host and run it as a service:
```bash
apt install firewalld
systemctl enable firewalld --now
```

Here, since 22 and 80 are used as docker forward ports, they should be open and under evaluation:
```bash
firewall-cmd --permanent --add-port=22/udp
firewall-cmd --permanent --add-port=22/tcp
firewall-cmd --permanent --add-port=80/udp
firewall-cmd --permanent --add-port=80/tcp
systemctl restart firewalld
firewall-cmd --runtime-to-permanent
```

The open ports can be listed using:
```bash
firewall-cmd --list-ports
```

# Multi-View


# Intrusion Detection System (IDS)
The IDS methods are implemented in "IDS" directory for all binary, 6 and 10 categorical classifiers. The file `AE.py` is implementation of AE in "IDS" which trains the AE and saves the checkpoint for later usages.

## Dataset
We have implemented and tested our proposed method on UNSW-NB15 dataset which is freely available on the internet and can be downloaded using `download.sh` bash code.

## pre-process
The dataset should be pre-processed to be fed into IDS. The pre-processing steps include loading data, converting non-numerical values to numerics using one-hot encoding, and normalizing. The code `preprocess.py` applies preprocessing which is imported in each neural network structure.

## IDS Models
The implemeted IDS models include:
- `CNN_AE.py`: CNN IDS model using AE extracted features as input.
- `CNNAtt_AE.py`: CNN-attention based IDS model using AE extracted features as input.
- `CNNAtt-balanced.py`: CNN-attention based IDS model using AE extracted features as input with balanced sampling of data for training.
- `CNNAttLstm_AE.py`: CNN-LSTM IDS model based on attention module using AE extracted features as input.