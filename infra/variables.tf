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
}

variable "domain_name" {
  description = "Domain name for the service"
  type        = string
  default     = "staging-ai-daedalus.govgig.us"
}

variable "route53_zone_id" {
  description = "Route53 hosted zone ID for domain_name (optional for DNS record creation)"
  type        = string
  default     = ""
}

variable "cors_origins" {
  description = "Allowed CORS origins passed to the container as JSON array"
  type        = list(string)
  default     = ["http://localhost:3000", "http://localhost:3001"]
}

variable "openai_api_key" {
  description = "OpenAI API key stored in env secret"
  type        = string
  sensitive   = true
}

variable "model_name" {
  description = "Primary model name (for reasoning/retrieval)"
  type        = string
  default     = "gpt-4o-mini"
}

variable "synthesizer_model" {
  description = "Synthesizer model name (for final response generation)"
  type        = string
  default     = "gpt-4o"
}

variable "embedding_model" {
  description = "Embedding model name"
  type        = string
  default     = "text-embedding-3-small"
}

variable "reranker_enabled" {
  description = "Whether to use LLM-based reranking"
  type        = bool
  default     = true
}

variable "reflection_threshold" {
  description = "Threshold for triggering self-healing reflection"
  type        = number
  default     = 0.50
}

variable "retrieval_top_k" {
  description = "Number of documents to retrieve initially"
  type        = number
  default     = 20
}

variable "dashboard_request_timeout" {
  description = "Timeout for dashboard requests in seconds"
  type        = number
  default     = 120
}

variable "debug_mode" {
  description = "Whether to enable debug logging/mode"
  type        = bool
  default     = false
}

variable "jwt_secret_key" {
  description = "JWT secret key used by API auth"
  type        = string
  sensitive   = true
}

variable "admin_api_key" {
  description = "Admin API key for operational endpoints"
  type        = string
  sensitive   = true
}

variable "cookie_secret" {
  description = "Cookie/session secret value"
  type        = string
  sensitive   = true
}

variable "min_capacity" {
  description = "Minimum ECS desired task count for autoscaling"
  type        = number
  default     = 1
}

variable "max_capacity" {
  description = "Maximum ECS desired task count for autoscaling"
  type        = number
  default     = 3
}

variable "target_cpu_utilization" {
  description = "Target average CPU utilization percent for ECS autoscaling"
  type        = number
  default     = 70
}

variable "target_memory_utilization" {
  description = "Target average memory utilization percent for ECS autoscaling"
  type        = number
  default     = 75
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
