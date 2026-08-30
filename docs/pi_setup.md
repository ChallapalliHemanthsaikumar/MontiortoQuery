# Raspberry Pi 4 Setup Guide

## 1. Enable the Camera

```bash
sudo raspi-config
```
Navigate to: **Interface Options → Camera → Enable**

Reboot:
```bash
sudo reboot
```

Test the camera:
```bash
libcamera-hello
```
You should see a 5-second preview window.

Take a test photo:
```bash
libcamera-jpeg -o test.jpg
```

## 2. Install System Dependencies

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv python3-opencv libatlas-base-dev
```

## 3. Set Up the Project

```bash
cd ~
git clone https://github.com/ChallapalliHemanthsaikumar/antvision.git
cd antvision
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 4. Install AWS IoT SDK

```bash
pip install awsiotsdk
```

## 5. Copy IoT Certificates

Place the certificate files (generated from AWS IoT Console) into:
```
~/antvision/certs/
├── device-certificate.pem.crt
├── private.pem.key
└── AmazonRootCA1.pem
```

## 6. Test the Pipeline

```bash
cd ~/antvision
python edge/main.py --live
```

## 7. Run on Boot (Optional)

Create a systemd service:
```bash
sudo nano /etc/systemd/system/antvision.service
```

```ini
[Unit]
Description=AntVision Edge Pipeline
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/antvision
ExecStart=/home/pi/antvision/venv/bin/python edge/main.py --live
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable:
```bash
sudo systemctl enable antvision
sudo systemctl start antvision
```
