# Map DEMPE function class numbers to categories
dempe_class_mapping = {
    "0": "Development",
    "1": "Enhancement",
    "2": "Maintenance",
    "3": "Protection",
    "4": "Exploitation",
    "Non-conventional": "Non-conventional",
}

# Friendly names for DEMPE classes
dempe_class_names = {
    "DEMPE_Class_0": "Development",
    "DEMPE_Class_1": "Enhancement",
    "DEMPE_Class_2": "Maintenance",
    "DEMPE_Class_3": "Protection",
    "DEMPE_Class_4": "Exploitation",
}


dempe_prediction_mapping = {
    0: "Development",
    1: "Enhancement",
    2: "Maintenance",
    3: "Protection",
    4: "Exploitation",
}

dempe_conv_commit_mapping = {
    "feat": [0],  # ✅ Development
    "perf": [1],  # ✅ Enhancement
    "breaking change": [1],  # ✅ Enhancement
    "fix": [2],  # ✅ Maintenance
    "chore": [2],  # ✅ Maintenance & Exploitation (context-sensitive)
    "docs": [2],  # ✅ Maintenance
    "style": [2],  # ✅ Maintenance
    "refactor": [2],  # ✅ Maintenance
    "test": [3],  # ✅ Protection
    "build": [4],  # ✅ Exploitation
    "ci": [4],  # ✅ Exploitation
}
