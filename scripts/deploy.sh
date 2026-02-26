#!/bin/bash
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Help Agent Deployment Script${NC}"
echo -e "${GREEN}========================================${NC}"

# Check required tools
command -v terraform >/dev/null 2>&1 || { echo -e "${RED}terraform required but not installed.${NC}" >&2; exit 1; }
command -v aws >/dev/null 2>&1 || { echo -e "${RED}aws cli required but not installed.${NC}" >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo -e "${RED}docker required but not installed.${NC}" >&2; exit 1; }

# Variables
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INFRA_DIR="$PROJECT_DIR/infra"
AWS_REGION="us-west-2"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="daedalus"
IMAGE_TAG="latest"

cd "$PROJECT_DIR"

# Get or prompt for DB password
if [ -z "$DB_PASSWORD" ]; then
    echo -e "${YELLOW}Enter database password:${NC}"
    read -s DB_PASSWORD
    echo ""
fi

echo -e "\n${GREEN}Step 1: Terraform Apply (RDS + Infrastructure)${NC}"
cd "$INFRA_DIR"
terraform init -upgrade
terraform apply -var="db_password=$DB_PASSWORD" -auto-approve

# Get outputs
RDS_ENDPOINT=$(terraform output -raw rds_endpoint)
RDS_PORT=$(terraform output -raw rds_port)
RDS_DATABASE=$(terraform output -raw rds_database)
ECR_URL=$(terraform output -raw ecr_repository_url)
ECS_CLUSTER=$(terraform output -raw ecs_cluster_name)
ECS_SERVICE=$(terraform output -raw ecs_service_name)

echo -e "${GREEN}RDS Endpoint: $RDS_ENDPOINT${NC}"

echo -e "\n${GREEN}Step 2: Wait for RDS to be available${NC}"
echo "Waiting for RDS instance to be available (this may take 5-10 minutes)..."
aws rds wait db-instance-available --db-instance-identifier daedalus-staging-pg --region $AWS_REGION
echo -e "${GREEN}RDS is ready!${NC}"

echo -e "\n${GREEN}Step 3: Build and push Docker image (includes data)${NC}"
cd "$PROJECT_DIR"

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URL

# Build image (includes pre-indexed data in data/ folder)
docker build -t $ECR_REPO:$IMAGE_TAG .

# Tag and push
docker tag $ECR_REPO:$IMAGE_TAG $ECR_URL:$IMAGE_TAG
docker push $ECR_URL:$IMAGE_TAG

echo -e "${GREEN}Docker image pushed!${NC}"

echo -e "\n${GREEN}Step 4: Update ECS service${NC}"
aws ecs update-service \
    --cluster $ECS_CLUSTER \
    --service $ECS_SERVICE \
    --force-new-deployment \
    --region $AWS_REGION

echo "Waiting for ECS deployment to stabilize..."
aws ecs wait services-stable \
    --cluster $ECS_CLUSTER \
    --services $ECS_SERVICE \
    --region $AWS_REGION

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}  Deployment Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Service URL: https://staging-ai-daedalus.govgig.us"
echo -e "RDS Endpoint: $RDS_ENDPOINT"
echo -e "\nTest with:"
echo -e "  curl 'https://staging-ai-daedalus.govgig.us/cai/api/chat/bot_response?query=What%20is%20FAR%2052.236-2'"
