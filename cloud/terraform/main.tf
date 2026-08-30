terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# ─── IoT Core ────────────────────────────────────────────────

resource "aws_iot_thing" "pi" {
  name = "${var.project_name}-pi01"
}

resource "aws_iot_policy" "device_policy" {
  name = "${var.project_name}-device-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["iot:Connect"]
        Resource = "arn:aws:iot:${var.aws_region}:*:client/${var.project_name}-*"
      },
      {
        Effect   = "Allow"
        Action   = ["iot:Publish"]
        Resource = "arn:aws:iot:${var.aws_region}:*:topic/${var.project_name}/events"
      }
    ]
  })
}

data "aws_iot_endpoint" "credentials" {
  endpoint_type = "iot:CredentialProvider"
}

resource "aws_iot_topic_rule" "events_to_lambda" {
  name        = "${var.project_name}_events"
  enabled     = true
  sql         = "SELECT * FROM '${var.project_name}/events'"
  sql_version = "2016-03-23"

  lambda {
    function_arn = aws_lambda_function.event_processor.arn
  }
}

# ─── DynamoDB ────────────────────────────────────────────────

resource "aws_dynamodb_table" "metrics" {
  name         = "${var.project_name}-metrics"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "experiment_id"
  range_key    = "timestamp"

  attribute {
    name = "experiment_id"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "S"
  }
}

# ─── S3 ──────────────────────────────────────────────────────

resource "aws_s3_bucket" "data" {
  bucket = "${var.project_name}-data-${var.environment}"
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ─── Lambda ──────────────────────────────────────────────────

data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../lambda"
  output_path = "${path.module}/lambda.zip"
}

resource "aws_lambda_function" "event_processor" {
  filename         = data.archive_file.lambda_zip.output_path
  function_name    = "${var.project_name}-event-processor"
  role             = aws_iam_role.lambda_role.arn
  handler          = "handler.lambda_handler"
  runtime          = "python3.11"
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  timeout          = 30

  environment {
    variables = {
      DYNAMODB_TABLE = aws_dynamodb_table.metrics.name
      S3_BUCKET      = aws_s3_bucket.data.id
    }
  }
}

resource "aws_lambda_permission" "iot_invoke" {
  statement_id  = "AllowIoTInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.event_processor.function_name
  principal     = "iot.amazonaws.com"
  source_arn    = aws_iot_topic_rule.events_to_lambda.arn
}

# ─── IoT Credentials Provider (Pi gets temp S3 creds via its certs) ───

resource "aws_iam_role" "device_role" {
  name = "${var.project_name}-device-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "credentials.iot.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "device_s3_policy" {
  name = "${var.project_name}-device-s3-policy"
  role = aws_iam_role.device_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject"
        ]
        Resource = "${aws_s3_bucket.data.arn}/*"
      }
    ]
  })
}

resource "aws_iot_role_alias" "device_alias" {
  alias               = "${var.project_name}-device-alias"
  role_arn             = aws_iam_role.device_role.arn
  credential_duration  = 3600
}

resource "aws_iot_policy" "credentials_policy" {
  name = "${var.project_name}-credentials-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["iot:AssumeRoleWithCertificate"]
        Resource = aws_iot_role_alias.device_alias.arn
      }
    ]
  })
}

# ─── Dashboard API (API Gateway + Lambda) ────────────────────

data "archive_file" "api_lambda_zip" {
  type        = "zip"
  source_file = "${path.module}/../lambda/api_handler.py"
  output_path = "${path.module}/api_lambda.zip"
}

resource "aws_lambda_function" "api" {
  filename         = data.archive_file.api_lambda_zip.output_path
  function_name    = "${var.project_name}-dashboard-api"
  role             = aws_iam_role.api_lambda_role.arn
  handler          = "api_handler.lambda_handler"
  runtime          = "python3.11"
  source_code_hash = data.archive_file.api_lambda_zip.output_base64sha256
  timeout          = 15

  environment {
    variables = {
      DYNAMODB_TABLE = aws_dynamodb_table.metrics.name
      S3_BUCKET      = aws_s3_bucket.data.id
    }
  }
}

resource "aws_apigatewayv2_api" "dashboard" {
  name          = "${var.project_name}-dashboard-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["GET", "OPTIONS"]
    allow_headers = ["Content-Type"]
  }
}

resource "aws_apigatewayv2_integration" "api" {
  api_id                 = aws_apigatewayv2_api.dashboard.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.api.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "api" {
  api_id    = aws_apigatewayv2_api.dashboard.id
  route_key = "GET /api/{proxy+}"
  target    = "integrations/${aws_apigatewayv2_integration.api.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.dashboard.id
  name        = "$default"
  auto_deploy = true
}

resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.dashboard.execution_arn}/*/*"
}

resource "aws_iam_role" "api_lambda_role" {
  name = "${var.project_name}-api-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "api_lambda_policy" {
  name = "${var.project_name}-api-lambda-policy"
  role = aws_iam_role.api_lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:Query",
          "dynamodb:GetItem"
        ]
        Resource = aws_dynamodb_table.metrics.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.data.arn,
          "${aws_s3_bucket.data.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# ─── IAM ─────────────────────────────────────────────────────

resource "aws_iam_role" "lambda_role" {
  name = "${var.project_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "lambda_policy" {
  name = "${var.project_name}-lambda-policy"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "dynamodb:PutItem",
          "dynamodb:BatchWriteItem"
        ]
        Resource = aws_dynamodb_table.metrics.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:PutObject"
        ]
        Resource = "${aws_s3_bucket.data.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}
