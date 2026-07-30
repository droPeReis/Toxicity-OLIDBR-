locals {
    name = "${var.name}-${var.environment}"

    tags = merge(
    {
        "Environment" = "${var.environment}"
    },
    var.tags
    )
}
