resource "aws_iam_role" "iam_role" {
  name = "${local.name}-role"

  # Change this to Fargate instead App Runner
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Principal = {
          Service = "build.apprunner.amazonaws.com"
        }
        Effect = "Allow"
      },
      {
        Action = "sts:AssumeRole"
        Principal = {
          Service = "tasks.apprunner.amazonaws.com"
        }
        Effect = "Allow"
      }
    ]
  })

  tags = merge(
    {
      Name = "${local.name}-role"
    },
    local.tags
  )
}


resource "aws_iam_role_policy" "bucket_policy" {
  name_prefix = "${local.name}-s3-policy"
  role        = aws_iam_role.iam_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = concat(
          aws_s3_bucket.bucket.*.arn,
        )
      },
      {
        Effect = "Allow"
        Action = [
          "s3:AbortMultipartUpload",
          "s3:GetBucketTagging",
          "s3:GetObject",
          "s3:GetObjectAcl",
          "s3:GetObjectTagging",
          "s3:GetObjectVersion",
          "s3:GetObjectVersionAcl",
          "s3:GetObjectVersionTagging",
          "s3:ListBucketMultipartUploads",
          "s3:ListMultipartUploadParts",
          "s3:PutBucketTagging",
          "s3:PutObject",
          "s3:PutObjectTagging",
          "s3:PutObjectVersionTagging",
          "s3:ReplicateTags",
        ]
        Resource = [
          for bucket in concat(aws_s3_bucket.bucket.*.arn) :
          "${bucket}/*"
        ]
      },
      {
        Sid    = "HTTPSOnly"
        Effect = "Deny"
        Action = "s3:*"
        Resource = concat(
          aws_s3_bucket.bucket.*.arn,
        )
        Condition = {
          Bool = {
            "aws:SecureTransport" = false
          }
        }
      }
    ]
  })
}
