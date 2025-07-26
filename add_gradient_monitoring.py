#!/usr/bin/env python3
"""Add gradient monitoring and stability checks to train_advanced.py"""

import re

# Read train_advanced.py
with open('train_advanced.py', 'r') as f:
    content = f.read()

# Add gradient monitoring function after imports
monitoring_code = '''
def compute_gradient_norm(model):
    """Compute the L2 norm of gradients."""
    total_norm = 0.0
    param_count = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    total_norm = total_norm ** 0.5
    return total_norm

def check_for_nan_gradients(model):
    """Check if any gradient contains NaN."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                return True, name
    return False, None
'''

# Insert after imports
import_end = content.find('\n\n', content.find('from typing'))
content = content[:import_end] + '\n' + monitoring_code + content[import_end:]

# Find the backward() call and add gradient checks
backward_pattern = r'(\s+)(loss\.backward\(\))'
replacement = r'''\1# Check for NaN before backward
\1if torch.isnan(loss):
\1    logger.warning(f"NaN loss detected at epoch {epoch}, batch {batch_idx}")
\1    logger.warning(f"Loss components: {loss_components}")
\1    # Skip this batch
\1    optimizer.zero_grad()
\1    continue
\1    
\1loss.backward()
\1
\1# Check gradients after backward
\1grad_norm_before_clip = compute_gradient_norm(model)
\1has_nan, nan_param = check_for_nan_gradients(model)
\1if has_nan:
\1    logger.warning(f"NaN gradient detected in {nan_param}")
\1    optimizer.zero_grad()
\1    continue'''

content = re.sub(backward_pattern, replacement, content)

# Add gradient norm to metrics logging
metrics_pattern = r'(metrics = \{[^}]+)\}'
replacement = r'''\1,
                        'grad_norm': grad_norm_before_clip
                    }'''
content = re.sub(metrics_pattern, replacement, content, flags=re.DOTALL)

# Save the modified file
with open('train_advanced.py', 'w') as f:
    f.write(content)

print("Added gradient monitoring to train_advanced.py")
print("Key additions:")
print("- compute_gradient_norm() function")
print("- check_for_nan_gradients() function")
print("- NaN loss detection before backward")
print("- NaN gradient detection after backward")
print("- Gradient norm logging in metrics")