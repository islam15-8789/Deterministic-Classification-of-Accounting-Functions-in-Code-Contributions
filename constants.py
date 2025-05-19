# Map DEMPE function class numbers to categories
dempe_class_mapping = {
    "0": "Development",
    "1": "Enhancement",
    "2": "Maintenance",
    "3": "Protection",
    "4": "Exploitation",
    "Non-conventional": "Non-conventional"
}


dempe_prediction_mapping = {
    0: "Development",
    1: "Enhancement",
    2: "Maintenance",
    3: "Protection",
    4: "Exploitation"
}

# Define the mapping of conventional commit types to DEMPE function classes
dempe_conv_commit_mapping = {
    "feat": 0,  # Development
    "perf": 1, "breaking change": 1, # Enhancement
    "fix": 2, "chore": 2, "docs": 2, "style": 2, "refactor": 2, # Maintenance
    "test": 3,  # Protection
    "build": 4, "ci": 4  # Exploitation
}