# Zero Trust Architecture Implementation Intrusion Detection System (ZTA-IDS)

## Overview
This project implements a Zero Trust Architecture (ZTA) for an Intrusion Detection System (IDS), focusing on simulating normal and attack requests through a web interface. It utilizes Docker to create a network of clients and servers, applies various IDS methods, and incorporates a modified multi-view approach for enhanced detection capabilities.

## Table of Contents
1. [Simulation of Requests](#simulation-of-requests)
2. [Server Setup](#server-setup)
3. [Client Setup](#client-setup)
4. [Network and Server Management](#network-and-server-management)
5. [Dataset for MV and IDS](#dataset)
6. [Multi-View Implementation](#multi-view-implementation)
7. [Intrusion Detection System](#intrusion-detection-system)
8. [Dataset and Preprocessing](#dataset-and-preprocessing)
9. [IDS Models](#ids-models)
10. [Real World Evaluation](#real-world-validation)
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

## Dataset
### UNSW-NB15 dataset

<a text = "link"> https://research.unsw.edu.au/projects/unsw-nb15-dataset</a>

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

## Real World Validation
Convert dumped pcap file to equivalent argus file using:
```bash
export filename=testmasscan; argus -r $filename.pcap -w $filename.argus
```
Then extract the attributes that are extractable by argus tools using:
```bash
export filename=testmasscan; ra -r $filename.argus -s saddr, daddr, sport, dur, proto, dport, state, spkts, dpkts, sbytes, dbytes, rate, sttl, dttl, sload, dload, sloss, dloss, sintpkt, dintpkt, sjit, djit, swin, stcpb, dtcpb, dwin, tcprtt, synack, ackdat, smeansz, dmeansz -c , > $filename.csv
```
The resulting $filename.csv contains most attributes of UNSW-NB15, however the others can be extracted by our implemented python code named "extract.py". 

`python3 extract.py testmasscan`

* Note: In the above codes, we used testmasscan for both argus and python running files

The resulting $filename.csv includes all attributes of UNSW-NB15, although it may have some additional columns due to one-hot encoding of non-numerical attributes with values other than known values of UNSW-NB15 dataset. We should remove or replace these values with most similart values. For example, according to our tests these values can be replaced as written bellow:

```bash
test_real_data.loc[test_real_data['state'] == 'URFIL', 'state'] = 'URN'
test_real_data.loc[test_real_data['state'] == 'STP', 'state'] = 'CLO'
test_real_data.loc[test_real_data['state'] == 'STA', 'state'] = 'PAR'
test_real_data.loc[test_real_data['state'] == 'NNS', 'state'] = 'no'
test_real_data.loc[test_real_data['state'] == 'URP', 'state'] = 'PAR'
test_real_data.loc[test_real_data['state'] == 'NRS', 'state'] = 'no'
test_real_data.loc[test_real_data['proto'] == 'man', 'proto'] = 'any'
test_real_data.loc[test_real_data['proto'] == 'ipv6-icmp', 'proto'] = 'icmp'
real_data = pd.DataFrame(ct.transform(test_real_data))
real_data.columns = new_cols
```

Also, do not forget to set any Nan value in the generated data:

```bash
real_data = real_data.replace(np.nan, 0)
```

real_data seems ready to be used by next modules.

### Requirements
!apt-get install argus-client