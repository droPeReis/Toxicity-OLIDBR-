variable "name" {
  description = "A name for the application."
  type = string
  default = "toxicity-detection"
}

variable "environment" {
  description = "Environment. It will be part of the application name and a tag in AWS Tags."
  type = string
  default = "dev"
}

variable "region" {
  description = "AWS Region name."
  type = string
  default = "us-east-1"
}

variable "tags" {
  type = map(string)
  description = "AWS Tags common to all the resources created."
  default = {}
}
