variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "staging"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "daedalus"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.30.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

variable "public_subnet_cidrs" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.30.1.0/24", "10.30.2.0/24", "10.30.3.0/24"]
}

variable "private_subnet_cidrs" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.30.11.0/24", "10.30.12.0/24", "10.30.13.0/24"]
}

variable "container_port" {
  description = "Container port"
  type        = number
  default     = 8000
}

variable "task_cpu" {
  description = "ECS task CPU units"
  type        = number
  default     = 1024
}

variable "task_memory" {
  description = "ECS task memory (MB)"
  type        = number
  default     = 2048
}

variable "desired_count" {
  description = "Desired number of ECS tasks"
  type        = number
  default     = 1
}

variable "acm_certificate_arn" {
  description = "ACM certificate ARN for HTTPS"
  type        = string
  default     = "arn:aws:acm:us-west-2:021891572695:certificate/38927297-28f6-44a8-a4af-f6451cf0d89c"
}

variable "domain_name" {
  description = "Domain name for the service"
  type        = string
  default     = "staging-ai-daedalus.govgig.us"
}

# RDS PostgreSQL
variable "db_instance_class" {
  description = "RDS instance class - r6g optimized for vector workloads"
  type        = string
  default     = "db.r6g.large"  # 2 vCPU, 8GB RAM - good for pgvector
}

variable "db_password" {
  description = "RDS master password"
  type        = string
  sensitive   = true
}
