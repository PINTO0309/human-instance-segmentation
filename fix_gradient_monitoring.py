#!/usr/bin/env python3
"""Fix the gradient monitoring implementation in train_advanced.py"""

import re

# Read train_advanced.py
with open('train_advanced.py', 'r') as f:
    content = f.read()

# Fix the loss_components reference
content = content.replace(
    'logger.warning(f"Loss components: {loss_components}")',
    'logger.warning(f"Loss components: {loss_dict}")'
)

# Fix the grad_norm_before_clip scope issue by initializing it
# Find the train_epoch function and add initialization
train_epoch_pattern = r'(def train_epoch\([^)]+\)[^:]+:\s*\n)((?:\s{4}.*\n)*?)(\s+# Training loop)'
def train_epoch_replacer(match):
    func_def = match.group(1)
    existing_vars = match.group(2)
    loop_comment = match.group(3)
    
    # Add grad_norm initialization
    new_vars = existing_vars + '    grad_norm_before_clip = 0.0  # Initialize for metrics\n'
    
    return func_def + new_vars + loop_comment

content = re.sub(train_epoch_pattern, train_epoch_replacer, content, flags=re.MULTILINE)

# Also need to update the metrics dict to handle the case when no batches were processed
metrics_pattern = r'(metrics = \{[^}]+)\'grad_norm\': grad_norm_before_clip\s*\}'
replacement = r"\1'grad_norm': grad_norm_before_clip if num_batches > 0 else 0.0\n    }"

content = re.sub(metrics_pattern, replacement, content, flags=re.DOTALL)

# Save the fixed file
with open('train_advanced.py', 'w') as f:
    f.write(content)

print("Fixed gradient monitoring issues:")
print("- Changed loss_components to loss_dict")
print("- Initialized grad_norm_before_clip = 0.0")
print("- Added check for num_batches > 0 in metrics")