variable "project_id" {
  description = "Google Cloud project ID (set in terraform.tfvars, which is git-ignored)."
  type        = string
}

variable "region" {
  description = "Region for the cluster and the image registry."
  type        = string
  default     = "us-east1"
}

variable "cluster_name" {
  description = "GKE Autopilot cluster name."
  type        = string
  default     = "brainct"
}

variable "repository_id" {
  description = "Artifact Registry repository for the API image."
  type        = string
  default     = "brainct"
}
