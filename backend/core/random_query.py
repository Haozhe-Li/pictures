from random import random


async def generate_random_query() -> str:
    choices = [
        "A serene landscape with mountains and a river",
        "A bustling city street at night with neon lights",
        "A close-up of a colorful butterfly on a flower",
        "A futuristic city skyline with flying cars",
        "A cozy cabin in the woods during autumn",
        "A majestic lion resting in the savannah",
        "A vibrant coral reef teeming with marine life",
        "A snowy mountain peak under a clear blue sky",
        "A group of friends enjoying a picnic in the park",
    ]
    return random.choice(choices)
