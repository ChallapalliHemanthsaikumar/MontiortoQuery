#!/bin/bash
# Generate IoT certificates for the Raspberry Pi
# Run this from your Windows machine with AWS CLI configured

set -e

THING_NAME="antvision-pi01"
POLICY_NAME="antvision-device-policy"
CERT_DIR="../certs"

mkdir -p "$CERT_DIR"

echo "Creating IoT certificate..."
CERT_ARN=$(aws iot create-keys-and-certificate \
    --set-as-active \
    --certificate-pem-outfile "$CERT_DIR/device-certificate.pem.crt" \
    --public-key-outfile "$CERT_DIR/public.pem.key" \
    --private-key-outfile "$CERT_DIR/private.pem.key" \
    --query 'certificateArn' \
    --output text)

echo "Certificate ARN: $CERT_ARN"

echo "Downloading Amazon Root CA..."
curl -o "$CERT_DIR/AmazonRootCA1.pem" https://www.amazontrust.com/repository/AmazonRootCA1.pem

echo "Attaching certificate to thing..."
aws iot attach-thing-principal \
    --thing-name "$THING_NAME" \
    --principal "$CERT_ARN"

echo "Attaching policy to certificate..."
aws iot attach-policy \
    --policy-name "$POLICY_NAME" \
    --target "$CERT_ARN"

ENDPOINT=$(aws iot describe-endpoint --endpoint-type iot:Data-ATS --query 'endpointAddress' --output text)
echo ""
echo "====================================="
echo "  IoT Setup Complete"
echo "====================================="
echo "Endpoint: $ENDPOINT"
echo "Certs in: $CERT_DIR/"
echo ""
echo "Copy certs to Pi:"
echo "  scp -r $CERT_DIR hemanth@<PI_IP>:~/antvision/"
echo ""
echo "Run pipeline with IoT:"
echo "  python edge/main.py --live --iot-endpoint $ENDPOINT"
