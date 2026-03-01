# Secrets Manager Secret
resource "aws_secretsmanager_secret" "env" {
  name        = "${var.project_name}/${var.environment}/env"
  description = "Environment variables for ${var.project_name} ${var.environment}"

  tags = {
    Name = "${var.project_name}-${var.environment}-env"
  }
}

resource "aws_secretsmanager_secret_version" "env" {
  secret_id = aws_secretsmanager_secret.env.id
  secret_string = jsonencode({
    OPENAI_API_KEY = var.openai_api_key
    MODEL_NAME     = var.model_name
  })
}

# Secret versions are managed by Terraform variables (recommended via CI/CD secrets).

# Auth secrets (JWT, admin key, cookie secret)
resource "aws_secretsmanager_secret" "auth" {
  name        = "${var.project_name}/${var.environment}/auth"
  description = "Auth secrets for ${var.project_name} ${var.environment}"

  tags = {
    Name = "${var.project_name}-${var.environment}-auth"
  }
}

resource "aws_secretsmanager_secret_version" "auth" {
  secret_id = aws_secretsmanager_secret.auth.id
  secret_string = jsonencode({
    JWT_SECRET_KEY = var.jwt_secret_key
    ADMIN_API_KEY  = var.admin_api_key
    COOKIE_SECRET  = var.cookie_secret
  })
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "main" {
  name              = "/ecs/${var.project_name}-${var.environment}"
  retention_in_days = 30

  tags = {
    Name = "${var.project_name}-${var.environment}-logs"
  }
}
