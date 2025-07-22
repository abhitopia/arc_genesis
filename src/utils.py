import math


def cosine_schedule(start_value: float, end_value: float, current_step: int, total_steps: int) -> float:
    """
    Cosine annealing schedule that smoothly transitions from start_value to end_value.
    
    Args:
        start_value: Initial value at step 0
        end_value: Final value at step total_steps
        current_step: Current training step
        total_steps: Total steps for the decay
    
    Returns:
        Interpolated value using cosine decay
    """
    if total_steps == 0:
        return start_value
    
    if current_step >= total_steps:
        return end_value
    
    # Cosine decay from start_value to end_value
    progress = current_step / total_steps  # 0 to 1
    cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))  # 1 to 0
    return end_value + (start_value - end_value) * cosine_factor
