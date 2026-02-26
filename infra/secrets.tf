# Secrets Manager Secret
resource "aws_secretsmanager_secret" "env" {
  name        = "${var.project_name}/${var.environment}/env"
  description = "Environment variables for ${var.project_name} ${var.environment}"

  tags = {
    Name = "${var.project_name}-${var.environment}-env"
  }
}

# Note: Secret values should be populated manually or via CI/CD
# aws secretsmanager put-secret-value --secret-id daedalus/staging/env --secret-string '{"MONGO_URI":"...", ...}'

# Auth secrets (JWT, admin key, cookie secret)
resource "aws_secretsmanager_secret" "auth" {
  name        = "${var.project_name}/${var.environment}/auth"
  description = "Auth secrets for ${var.project_name} ${var.environment}"

  tags = {
    Name = "${var.project_name}-${var.environment}-auth"
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "main" {
  name              = "/ecs/${var.project_name}-${var.environment}"
  retention_in_days = 30

  tags = {
    Name = "${var.project_name}-${var.environment}-logs"
  }
}
