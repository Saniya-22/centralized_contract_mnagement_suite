# RDS PostgreSQL with pgvector for Help Agent RAG
# Optimized for low latency vector search

# DB Subnet Group (uses existing private subnets)
resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-${var.environment}-db-subnet"
  subnet_ids = aws_subnet.private[*].id

  tags = {
    Name = "${var.project_name}-${var.environment}-db-subnet"
  }
}

# Security Group for RDS
resource "aws_security_group" "rds" {
  name        = "${var.project_name}-${var.environment}-rds-sg"
  description = "Security group for RDS PostgreSQL"
  vpc_id      = aws_vpc.main.id

  # Allow PostgreSQL from ECS tasks only
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
    description     = "PostgreSQL from ECS"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-rds-sg"
  }
}

# RDS Parameter Group optimized for vector workloads
resource "aws_db_parameter_group" "main" {
  family = "postgres16"
  name   = "${var.project_name}-${var.environment}-pg16-vector"

  # Static params require pending-reboot
  parameter {
    name         = "shared_preload_libraries"
    value        = "pg_stat_statements"
    apply_method = "pending-reboot"
  }

  # Dynamic params can be immediate
  parameter {
    name  = "maintenance_work_mem"
    value = "524288"  # 512MB for index building
  }

  parameter {
    name  = "effective_cache_size"
    value = "6291456"  # 6GB - r6g.large has 8GB RAM
  }

  parameter {
    name  = "work_mem"
    value = "131072"  # 128MB per operation
  }

  parameter {
    name  = "random_page_cost"
    value = "1.1"  # SSD optimized
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-pg-params"
  }
}

# RDS PostgreSQL Instance
resource "aws_db_instance" "main" {
  identifier = "${var.project_name}-${var.environment}-pg"

  # Engine
  engine               = "postgres"
  engine_version       = "16.6"
  instance_class       = var.db_instance_class

  # Storage - gp3 with baseline performance (fast enough for 30MB data)
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp3"
  # gp3 baseline: 3000 IOPS, 125 MB/s - plenty for our use case

  # Database
  db_name  = "daedalus"
  username = "daedalus_admin"
  password = var.db_password
  port     = 5432

  # Network - same VPC as ECS for lowest latency
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  publicly_accessible    = false
  multi_az               = var.environment == "production" ? true : false

  # Parameters
  parameter_group_name = aws_db_parameter_group.main.name

  # Backup
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "Mon:04:00-Mon:05:00"

  # Performance Insights for monitoring
  performance_insights_enabled          = true
  performance_insights_retention_period = 7

  # Other
  auto_minor_version_upgrade = true
  deletion_protection        = var.environment == "production" ? true : false
  skip_final_snapshot        = var.environment != "production"
  final_snapshot_identifier  = var.environment == "production" ? "${var.project_name}-${var.environment}-final-snapshot" : null

  tags = {
    Name = "${var.project_name}-${var.environment}-pg"
  }
}

# Store DB credentials in Secrets Manager
resource "aws_secretsmanager_secret" "db" {
  name        = "${var.project_name}/${var.environment}/db"
  description = "RDS PostgreSQL credentials for ${var.project_name}"

  tags = {
    Name = "${var.project_name}-${var.environment}-db-secret"
  }
}

# Secret keys (host, port, database, username, password) are referenced by ECS task definition in ecs.tf.
# Keep in sync: container secrets valueFrom uses :key:: for each key below.
resource "aws_secretsmanager_secret_version" "db" {
  secret_id = aws_secretsmanager_secret.db.id
  secret_string = jsonencode({
    host     = aws_db_instance.main.address
    port     = tostring(aws_db_instance.main.port)
    database = aws_db_instance.main.db_name
    username = aws_db_instance.main.username
    password = var.db_password
  })
}
