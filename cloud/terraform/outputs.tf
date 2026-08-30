output "iot_thing_name" {
  value = aws_iot_thing.pi.name
}

output "dynamodb_table" {
  value = aws_dynamodb_table.metrics.name
}

output "s3_bucket" {
  value = aws_s3_bucket.data.id
}

output "lambda_function" {
  value = aws_lambda_function.event_processor.function_name
}

output "iot_topic" {
  value = "${var.project_name}/events"
}

output "iot_role_alias" {
  value = aws_iot_role_alias.device_alias.alias
}

output "iot_credentials_endpoint" {
  value = data.aws_iot_endpoint.credentials.endpoint_address
}

