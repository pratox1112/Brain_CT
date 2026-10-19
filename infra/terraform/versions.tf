terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  # State is kept locally (terraform.tfstate, git-ignored). Fine for one person; a team would
  # use a remote backend such as a GCS bucket.
}

provider "google" {
  project = var.project_id
  region  = var.region
}
