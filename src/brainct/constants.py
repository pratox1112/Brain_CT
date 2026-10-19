"""Constants shared by training and serving (kept dependency-free so the API image can use them)."""

# The model's 6 outputs, in this order: any hemorrhage, then the 5 subtypes.
LABELS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
SUBTYPES = LABELS[1:]

DISCLAIMER = "Research use only. Not a medical device; not for clinical decisions or diagnosis."
