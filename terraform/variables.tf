variable "aws_region" {
  description = "The AWS region to create resources in"
  default     = "us-west-2"
}

variable "ami_id" {
  description = "The AMI ID for the EC2 instance"
  default     = "ami-0c55b159cbfafe1f0"  # Amazon Linux 2 AMI (HVM), SSD Volume Type
}

variable "db_username" {
  description = "Username for the RDS instance"
}

variable "db_password" {
  description = "Password for the RDS instance"
}