<p align="center">
 <a href="https://github.com/dav354/seminar/actions">
    <img src="https://github.com/dav354/seminar/actions/workflows/docker_build.yml/badge.svg?branch=main"
         alt="Build Status" />
  </a>
  <br/>
  <br/>
  <img src="assets/logo_white.png"  
    alt="Logo"
    width="400" />
  <br/>
  <br/>
</p>
<h1 align="center">Rock Pepper Scissors</h1>
  <br/>

This repo contains the code for our seminar project. The goal is detect some hand gestures via the internal camera of the [pepper robot](https://aldebaran.com/en/). And to play the rock paper scissors game against it.

## Table of Contents

* [Overview](#overview)
  * [Hardware](#hardware)
  * [Software](#software)
    * [Game Server](#game-server)
    * [Pepper API](#pepper-api)
* [Setup](#setup)
  * [Venv](#venv)
  * [Install edge tpu repo](#install-edge-tpu-repo)
  * [Pi installation with ansible](#pi-installation-with-ansible)
  * [Manual Build](#manual-build)
  * [PI Wifi Cli](#pi-wifi-cli)

# Overview

## Hardware

To run this setup, you will need the following parts:
- Pepper Robot (v2.9)
- Raspberry Pi (We use the Pi5 with 8G Ram, which is overkill):
  - SD-Card
  - Powersupply
  - Google Coral Tpu USB
  - RPi Cooler

And you need to ensure to connect both, the Pi and the Pepper to the same Network.

## Software

This setup constists of three parts:

- **Game Server**

  This is the main logic handling the compute and game logic

- **Pepper API**

  This script is running in the head of the pepper robot and provides the API for the camera and movements.

- **Model Training**

  [This repo](https://github.com/dav354/model_training) contains the training script and data to build the custom ai model

### Game Server

The following environment variables are available. Set via the [compose.yml](compose.yml):

|env|format|description|
|:-:|-|-|
|`PEPPER_IP`|`http://ip-address:port`|The Ip address of the remote Pepper robot which runs the api.|

### Pepper Api

This API allows you to interact with a Pepper robot for camera streaming, gesture control, and speech (TTS and ASR). All endpoints are served over HTTP.

### **Endpoints**

#### **1. Video Stream**

* **GET `/video_feed`**

  Returns an MJPEG video stream from the robot’s camera.

#### **2. Gestures**

* **GET `/gesture/{rock, paper, scissors, swing}`**

  Makes Pepper perform a gesture.


# Setup

## Venv

To develop you need to setup **python3.10** and create the venv.

```shell
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Install edge tpu repo

Then you need to ensure you have the coral tpu dependencies installed.

```shell
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install edgetpu-compiler
```

## Pi installation with ansible

To setup the pi there is a small ansbile script. To run it you need to install ansible.

> [!NOTE]
> This step assumes you installed ubuntu24.04

1. First install the required galaxy roles:
  
  ```shell
  ansible-galaxy install geerlingguy.docker
  ```

2. Now you can execute the the playbooK:

  ```shell
  ansible-playbook -i '192.168.0.189,' -u david setup_pi.yml -Kk --diff
  ```

## Manual Build

to build the docker image locally run:

```shell
docker buildx create --name multiarch-builder --driver docker-container --use
docker buildx build \
  --builder multiarch-builder \
  --platform linux/arm64 \
  --load \
  -t rps:latest \
  game_server
```

### PI Wifi Cli

```bash
# sudo nano /etc/netplan/01-wlan.yaml

network:
  version: 2
  wifis:
    wlan0:
      optional: true
      access-points:
        "HUAWEI-0100T4":
          password: "YourPassword"
      dhcp4: true
```

and then apply the config

```bash
sudo chmod 600 /etc/netplan/01-wlan.yaml
sudo netplan apply
```