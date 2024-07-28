provider "aws" {
  region = var.aws_region
}

module "vpc" {
  source = "terraform-aws-modules/vpc/aws"

  name = "ai-trading-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true

  tags = {
    Terraform   = "true"
    Environment = "dev"
  }
}

resource "aws_instance" "app_server" {
  ami           = var.ami_id
  instance_type = "t2.micro"
  subnet_id     = module.vpc.public_subnets[0]

  tags = {
    Name = "AI Trading App Server"
  }
}

resource "aws_db_instance" "default" {
  allocated_storage    = 20
  storage_type         = "gp2"
  engine               = "postgres"
  engine_version       = "13.7"
  instance_class       = "db.t3.micro"
  name                 = "ai_trading"
  username             = var.db_username
  password             = var.db_password
  parameter_group_name = "default.postgres13"
  skip_final_snapshot  = true
  subnet_id            = module.vpc.private_subnets[0]

  tags = {
    Name = "AI Trading Database"
  }
}