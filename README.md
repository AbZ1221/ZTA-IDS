# Zero Trust Architecture Implementation Intrusion Detection System (ZTA-IDS)

## Overview
This project implements a Zero Trust Architecture (ZTA) for an Intrusion Detection System (IDS), focusing on simulating normal and attack requests through a web interface. It utilizes Docker to create a network of clients and servers, applies various IDS methods, and incorporates a modified multi-view approach for enhanced detection capabilities.

## Table of Contents
1. [Simulation of Requests](#simulation-of-requests)
2. [Server Setup](#server-setup)
3. [Client Setup](#client-setup)
4. [Network and Server Management](#network-and-server-management)
5. [Multi-View Implementation](#multi-view-implementation)
6. [Intrusion Detection System](#intrusion-detection-system)
7. [Dataset and Preprocessing](#dataset-and-preprocessing)
8. [IDS Models](#ids-models)

---

## Simulation of Requests
To simulate web requests, we implement a website on a server to sniff all incoming requests. We use Docker and docker-compose to create a network of clients and servers with a range of IP addresses.

## Server Setup
### Docker Configuration for Server
- Pull the Docker image: `docker pull ubuntu:20.04`
- Create and configure the Docker container:
  ```bash
  docker run --rm -it --name ubuntu ubuntu:20.04 bash
  # Followed by installation commands

### Server Modules Installation
Update and install necessary modules:

```bash
apt update
apt install build-essential git openssh-server -y
# Additional installation commands
```

### Finalizing Server Setup
Copy SSH config to the container and commit changes:

```bash
docker cp sshd_config ubuntu:/
docker commit ubuntu gitea
```

## Client Setup
### Docker Configuration for Client
We use Kali Linux Docker image for client setup.

Build the client Docker image:

```bash
cd Client
docker build -t kalilinux/kali-rolling:latest .
cd ..
```

## Network and Server Management
### Running Server and Network
Edit docker-compose.yml to set the desired IP range.

Start the server and network:
```bash
docker-compose up
```

SSH Service and Network Monitoring

Manually start the SSH service and run tcpdump for network monitoring.

## Multi-View Implementation
### Requirements

    Install dependencies: pip install pycrypto

### Running Multi-View
Place the UNSW-NB15 CSV files in the MV directory.

Execute the script: `python3 mv.py`

## Intrusion Detection System
### Dataset
The UNSW-NB15 dataset is used, downloadable via `download.sh`.

### Pre-processing
Pre-process the dataset using preprocess.py.

### IDS Models
Various IDS models implemented in Python:
- CNN with Autoencoder features (CNN_AE.py)
- CNN-Attention with Autoencoder features (CNNAtt_AE.py)
- CNN-Attention with balanced data sampling (CNNAtt-balanced.py)
- CNN-LSTM with Attention module (CNNAttLstm_AE.py)
