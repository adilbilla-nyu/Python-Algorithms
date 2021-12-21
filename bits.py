from math import log2, floor
from typing import List, Tuple, Dict
from functools import reduce
from operator import xor
from itertools import count

'''

Given an integer N. The task is to return the position of the first
set bit from the right side in the binary representation of the number

'''
def first_set_bit_pos(n: int) -> int:
    # n & ~(n - 1) extracts the first set bit
    # taking the log base 2 of the resulting value
    # gives us the zero based position of that set bit
    # we add 1 to account for the problem statement.
    return floor( log2(n & ~(n - 1)) ) + 1 if n > 0 else 0

'''

Given two numbers M and N. The task is to find the rightmost different bit
in the binary representation of the numbers

'''
def rightmost_different_bit_pos(n: int, m: int) -> int:
    xor_mask = n ^ m
    return floor( log2(xor_mask & ~(xor_mask - 1)) ) + 1 if xor_mask > 0 else 0

'''

Given a non-negative integer N. The task is to check if N is a power of 2.
More formally, check if N can be expressed as 2 ^ x for some x.

'''
def is_power_of_2(n: int) -> bool:
    # a number is a power of 2, if it has exactly 1 set bit
    return n != 0 and (n & (n - 1)) == 0

'''

Given a number N and a value K. From the right, set the Kth bit in the
binary representation of N. The position of Least Significant Bit (or last bit)
is 0, the second to last bit is 1 and so on.

'''
def set_kth_bit(n: int, k: int) -> int:
    return n | (1 << k)

'''

in a party of N people, each person is denoted by an integer. Couples
are represented by the same number. Find out the only single person
in the party of couples.

'''
def single_person(party: List[int]) -> int:
    return reduce(xor, party)

'''

Given a number N. The task is to check whether it is sparse or not. A number
is said to be a sparse number if no two or more consecutive bits are set
in the binary representation.

'''

def is_sparse(n: int) -> bool:
    return n & (n << 1) == 0

'''

You are given two numbers A and B. The task is to count the number of bits
needed to be flipped to convert A to B.

'''
def bit_difference(a: int, b: int) -> int:
    xor_mask = a ^ b

    for c in count(0):
        if xor_mask == 0:
            return c
        
        xor_mask &= xor_mask - 1

'''

Given an unsigned integer N. The task is to swap all odd bits with even bits.
For example, if the given number is 23 (00010111), it should be converted to
43(00101011). Here every even position bit is swapped with adjacent bit on the
right side. Every odd position bit is swapped with an adjacent on the left
side. Assume, we are dealing with 32 bit integers.

'''
def swap_even_odd_bits(n: int) -> int:
    # extract all even bits
    even_bits = n & 0x55555555

    # extract all odd bits
    odd_bits = n & 0xAAAAAAAA

    # move all even bits to odd position and move all odd bits to even position
    return (even_bits << 1) | (odd_bits >> 1)

'''

Given an integer N and an integer D, rotate the binary representation of
the integer N by D digits to the left as well as right and print the results
in decimal values after each of the rotation. Integer N is stored using
16 bits.

'''
def rotate_bits(n: int, d: int) -> Tuple[int, int]:
    return (n << d) | ( (n >> 16 - d) & ( (1 << 16 - d) - 1) ), \
           (n >> d) | ((n & ((1 << d) - 1)) << 16 - d)

'''

Given a 32 bit number X, reverse its binary form and print the answer
in decimal.

'''
def reverse_bits(x: int) -> int:
    for bit_position in range(16):
        if x & (1 << bit_position) != x & (1 << 31 - bit_position):
            x ^= (1 << bit_position)
            x ^= (1 << 31 - bit_position)
    
    return x

'''

Given a positive integer N, print count of set bits in it.

'''
def count_set_bits(n: int) -> int:
    for c in count(0):
        if n == 0: return c
        n &= n - 1

'''

Given a number N and a bit number K, check if kth bit of N is set or not.
A bit is called set if it is 1. Position of set bit '1' should be indexed
starting with 0 from LSB side in binary representation of the number.

'''
def kth_bit_is_set(n: int, k: int) -> bool:
    return n & (1 << k) > 0

'''

Given a number N. Find the length of the longest consecutive 1s in its
binary representation.

'''
def longest_consecutive_ones(n: int) -> int:
    for c in count(0):
        if n == 0: return c
        n &= (n >> 1)

'''

Given a sorted array A[] of N positive integers having all the numbers
occuring exactly twice, except for one number which will occur only once.
Find the number occurring only once.

'''
def occurs_once(A: List[int]) -> int:
    return reduce(xor, A)

'''

Given two integers a and b. Find the sum of two numbers without using
arithmetic operators. a >= 0, b >= 0

'''
def sum_(a: int, b: int) -> int:
    # the idea is to compute the carry bits using the & operator and
    # the sum bits using the xor (^) operator. After the first iteration
    # we have the sum of a and b without the carry bits. We have to add those
    # carry bits back into a (which stores the sum) and so we continue as long
    # as there is a carry.

    while b != 0:

        carry = a & b
        a ^= b
        b = carry << 1
    
    return a

'''

Given a number N, generate bit patterns from 0 to 2^n - 1 such that
successive patterns differ by one bit. A gray code sequence must begin with 0.

'''

def print_binary(n: int) -> str:
    # find position of most significant bit
    msb_position = floor( log2(n) )

    binary_representation = ''
    while msb_position >= 0:
        binary_representation += str((n >> msb_position) & 0x1)
        msb_position -= 1
    
    return binary_representation

def gray_code_patterns(n: int):
    '''
        suppose we have some number with the binary representation b1 b2 b3 b4.
        The corresponding gray code is defined to be g1 g2 g3 g4 where g1 = b1,
        g2 = b1 ^ b2, g3 = b2 ^ b3, g4 = b3 ^ b4. Generally g_n = b_(n - 1) ^ b_n

        The gray code of some number n, can easily be computed with a single expression:
        n ^ (n >> 1)
    
    '''
    output =  [print_binary(code ^ (code >> 1)) for code in range(1, 2 ** n)]

    print('0' * n, end=' ')
    for gray_code in output:
        print(f'{gray_code:0{n}}', end=' ')

'''

Given an arr[] of N positive elements. The task is to find the Maximum AND value
generated by any pair of elements from the array.

'''
def max_and(elements: List[int]) -> int:
    pattern_count = lambda elements, pattern: sum(1 for element in elements if element & pattern == pattern)
    pattern = 0

    for bit_position in range(31, -1, -1):
        if pattern_count(elements, pattern | 1 << bit_position) > 1:
            pattern |= (1 << bit_position)
    
    return pattern

'''

Given an array arr[] of N positive integers. Find an integer denoting the maximum
XOR subset value in the given array.

'''
def max_xor(elements: List[int]) -> int:

    # this algorithm implements gaussian elimination. The basic idea is that
    # we iterate through all buts 31 to 0, inclusive, and we check if that bit is
    # set, if it is set, we move maximum of all elements with that set bit to the
    # front of the array at index `index` which is initialized to 0. We then
    # "eliminate" all other elements with that bit set by xoring it with
    # the maximum element. We then repeat the process for each bit.

    index = 0
    for bit_position in range(31, -1, -1):
        max_index, max_element = index, -1
        for i in range(index, len(elements)):
            if kth_bit_is_set(elements[i], bit_position) and elements[i] > max_element:
                max_index, max_element = i, elements[i]
        
        if max_element == -1: continue
        
        elements[index], elements[max_index] = elements[max_index], elements[index]
        max_index = index

        for i, _ in enumerate(elements):
            if kth_bit_is_set(elements[i], bit_position) and i != max_index:
                elements[i] ^= elements[max_index]
        
        index += 1
    
    return reduce(xor, elements)

'''

We define f(X, Y) as number of different corresponding bits in binary
representation of X and Y. For example, f(2, 7) = 2, since binary
representation of 2 and 7 are 010 and 111, respectively. The first
and third bit differ, so f(2, 7) = 2.

You are given an array A of N integers, A_1, A_2, ..., A_N. Find the sum
of f(A_i, A_j) for all ordered pairs (i, j) such that 1 <= i, j <= N.
Return answer modulo 10E9 + 7.

'''
def count_bits(elements: List[int]) -> int:
    result = 0
    for bit_position in range(32):

        # count number of set bits at position `bit_position`
        set_bits = sum(1 for element in elements if element & (1 << bit_position) > 0)

        # number of elements with bit position not at at `bit_position`
        unset_bits = len(elements) - set_bits

        result += set_bits * unset_bits * 2
    
    return result % int(10E9 + 7)

'''

Given a positive integer n, count the total number of set bits in
binary representation of all numbers from 1 to n.

'''

# preprocessing (assuming 32 bit integers)
def generate_bit_count_table(number_of_bits: int) -> Dict[int, int]:
    lookup_table = {0: 0}
    for element in range(number_of_bits + 1):
        lookup_table[element] = lookup_table[element >> 1] + (element & 0x1)

    return lookup_table

'''
look_up_table = generate_bit_count_table(256)
def count_set_bits_lookup_table(n: int) -> int:
    return look_up_table[n & 0xFF] + look_up_table[(n >> 8) & 0xFF] + look_up_table[(n >> 16) & 0xFF] + look_up_table[(n >> 24) & 0xFF]
'''

def count_total_set_bits(n: int) -> int:
    pass    