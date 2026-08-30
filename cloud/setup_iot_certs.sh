#!/bin/bash
# Generate IoT certificates for the Raspberry Pi
# Run this from the cloud/ directory on your Windows machine

set -e

THING_NAME="antvision-pi01"
DEVICE_POLICY="antvision-device-policy"
CRED_POLICY="antvision-credentials-policy"
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

echo "Attaching device policy (MQTT)..."
aws iot attach-policy \
    --policy-name "$DEVICE_POLICY" \
    --target "$CERT_ARN"

echo "Attaching credentials policy (S3 via IoT)..."
aws iot attach-policy \
    --policy-name "$CRED_POLICY" \
    --target "$CERT_ARN"

MQTT_ENDPOINT=$(aws iot describe-endpoint --endpoint-type iot:Data-ATS --query 'endpointAddress' --output text)
CRED_ENDPOINT=$(aws iot describe-endpoint --endpoint-type iot:CredentialProvider --query 'endpointAddress' --output text)

echo ""
echo "====================================="
echo "  IoT Setup Complete"
echo "====================================="
echo ""
echo "MQTT Endpoint:        $MQTT_ENDPOINT"
echo "Credentials Endpoint: $CRED_ENDPOINT"
echo "Certs in:             $CERT_DIR/"
echo ""
echo "Copy certs to Pi:"
echo "  scp -r certs hemanth@<PI_IP>:~/antvision/"
echo ""
echo "Run pipeline with IoT + S3 (no stored keys!):"
echo "  python edge/main.py --live \\"
echo "    --iot-endpoint $MQTT_ENDPOINT \\"
echo "    --s3-bucket antvision-data-dev \\"
echo "    --credentials-endpoint $CRED_ENDPOINT \\"
echo "    --role-alias antvision-device-alias"
