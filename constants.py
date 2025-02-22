# Map DEMPE function class numbers to categories
dempe_class_mapping = {
    "0": "Development",
    "1": "Enhancement",
    "2": "Maintenance",
    "3": "Protection",
    "4": "Exploitation",
    "Non-conventional": "Non-conventional"
}

# Define the mapping of conventional commit types to DEMPE function classes
dempe_conv_commit_mapping = {
    "feat": 0, "breaking change": 0,  # Development
    "refactor": 1, "perf": 1,  # Enhancement
    "fix": 2, "chore": 2, "ci": 2, "docs": 2, "style": 2, "test": 2,  # Maintenance
    "fix": 3, "chore": 3, "test": 3, "docs": 3,  # Protection
    "build": 4, "ci": 4  # Exploitation
}