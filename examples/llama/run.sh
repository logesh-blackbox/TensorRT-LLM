import math

def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    sqrt_n = math.isqrt(n)
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True

def find_prime_numbers(max_num):
    prime_numbers = []
    for num in range(3, max_num + 1, 2):
        if is_prime(num):
            prime_numbers.append(num)
    return prime_numbers

text="Write a program to find prime numbers that are odd"


python run.py --max_output_len=256 --tokenizer_dir=glaiveai/glaive-coder-7b  --engine_dir=glaive_multitask_tensorrt --input_text="${text}"

