variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "antvision"
}

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "dev"
}
