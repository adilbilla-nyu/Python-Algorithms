from typing import List
from string import ascii_lowercase


# utility function to check if input array is sorted
def is_sorted(elements: List[int]) -> None:
    return all(elements[i - 1] <= elements[i] for i in range(1, len(elements)))

'''

Given an integer N and a list arr. Sort the array using bubble sort algorithm.

'''
def bubble_sort(elements: List[int]) -> None:
    for i, _ in enumerate(elements):
        is_sorted = True
        for j in range(1, len(elements) - i):
            if elements[j] < elements[j - 1]:
                elements[j], elements[j - 1] = elements[j - 1], elements[j]
                is_sorted = False
        
        if is_sorted: break

'''

Given an integer N and a list arr. Sort the array using the insertion sort algorithm.

'''
def insertion_sort(elements: List[int]) -> None:
    for i in range(1, len(elements)):
        unsorted_element = elements[i]
        j = i - 1
        while j >= 0 and elements[j] > unsorted_element:
            elements[j + 1] = elements[j]
            j -= 1
        
        elements[j + 1] = unsorted_element

'''

Given an unsorted array of size N, use selection sort to sort arr[]
in increasing order.

'''
def selection_sort(elements: List[int]) -> None:
    sorted_index = 0
    for i in range(len(elements)):
        # get the min index from [i, n]
        min_index = i
        for j in range(i, len(elements)):
            if elements[j] < elements[min_index]:
                min_index = j
        
        elements[min_index], elements[sorted_index] = elements[sorted_index], elements[min_index]
        sorted_index += 1

'''

Given an array arr[], its starting position l and its ending position r.
Sort the array using merge sort algorithm.

'''
def merge(elements: List[int], l: int, m: int, r: int) -> None:
    left_index, right_index = l, m + 1

    # left_index <= m is used here because the elements at index m
    # is included in the left half. m + 1 to r (inclusive) is included
    # in the right half.

    temp = []
    while left_index <= m and right_index <= r:
        if elements[left_index] < elements[right_index]:
            temp.append(elements[left_index])
            left_index = left_index + 1
        else:
            temp.append(elements[right_index])
            right_index = right_index + 1
    
    while left_index <= m:
        temp.append(elements[left_index])
        left_index = left_index + 1
    
    while right_index <= r:
        temp.append(elements[right_index])
        right_index = right_index + 1
    
    for i in range(l, r + 1):
        elements[i] = temp[i - l]

def merge_sort(elements: List[int], l: int, r: int) -> None:
    if l < r:
        m = l + (r - l) // 2
        # sort the left half of the array
        merge_sort(elements, l, m)

        # sort the right half of the array
        merge_sort(elements, m + 1, r)

        # merge the sorted halves
        merge(elements, l, m, r)

'''

QuickSort is a Divide and Conquer algorithm. It picks an element as pivot
and partitions the given array around the picked pivot.

Given an array arr[], its starting position low and its ending position high.

implement the partition() and quickSort() functions to sort the array.

'''

# lomuto partition
def partition(elements: List[int], low: int, high: int) -> int:
    # choose the last element as the pivot
    pivot = elements[high]

    # index used for swapping
    pivot_index = low
    # iterate through the array from low to high - 1 and if the element is
    # less than or equal to the pivot place it to left by swapping it with
    # pivot index. All elements to the left and including the pivot_index
    # are less than or equal to the pivot. pivot_index will store the index
    # of the pivot in it's correct position at the end.
    for index in range(low, high):
        if elements[index] <= pivot:
            elements[index], elements[pivot_index] = elements[pivot_index], elements[index]
            pivot_index = pivot_index + 1
    
    # pivot index stores the last element past the lower side
    # so this is where the actual pivot should be placed
    elements[pivot_index], elements[high] = elements[high], elements[pivot_index]
    return pivot_index

def dutch_national_flag(elements: List[int]) -> None:
    index, lower_bound = 0, 0
    while index < len(elements):
        if elements[index] == 0:
            elements[index], elements[lower_bound] = elements[lower_bound], elements[index]
            lower_bound = lower_bound + 1
        index = index + 1
    
    index = lower_bound
    upper_bound = len(elements) - 1
    while index < upper_bound:
        if elements[index] == 2:
            elements[index], elements[upper_bound] = elements[upper_bound], elements[index]
            upper_bound = upper_bound - 1
        index = index + 1      

def quick_sort(elements: List[int], low: int, high: int) -> None:
    if low < high:
        partition_point = partition(elements, low, high)
        quick_sort(elements, low, partition_point - 1)
        quick_sort(elements, partition_point + 1, high)

'''

Heap Sort
Given an array of size N. The task is to sort the array elements by
completing the functions heapify() and build_heap() which are used to
implement heap_sort()

'''

# we must pass in heap_length into heapify() as heap_sort() works
# by incrementally making the heap smaller and smaller as the elements
# are put into their correct place. For example, in heap sort we take the
# root (largest element) and place it at the end of the array and decrease
# the heap size since it's no longer apart of the heap. when we heapify the
# root element, we'll constantly be taking the same max and bringing it back
# to the root
def heapify(elements: List[int], heap_length: int,  index: int) -> None:
    left_child, right_child = 2 * index + 1, 2 * index + 2
    largest = index

    if left_child < heap_length and elements[largest] < elements[left_child]:
        largest = left_child
    
    if right_child < heap_length and elements[largest] < elements[right_child]:
        largest = right_child
    
    if largest != index:
        elements[index], elements[largest] = elements[largest], elements[index]
        heapify(elements, heap_length, largest)

def build_heap(elements: List[int]) -> None:
    for index in range(len(elements) // 2, -1, -1):
        heapify(elements, len(elements), index)

def heap_sort(elements: List[int]) -> None:
    build_heap(elements)
    # since this is a max heap, we know that the root
    # which is index 0, contains the max element in the
    # array, we swap that into the last position of the array
    # hence, placing the max element in it's correct position.
    # however, this breaks the max heap property of the array
    # therefore, we heapify once again, at the root so that the
    # max element is once again at index 0
    for index in range(len(elements) - 1, 0, -1):
        elements[index], elements[0] = elements[0], elements[index]
        # heapify root element
        heapify(elements, index, 0)

'''

Given a string arr consisting of lowercase english letters, arrange all
its letters in lexicographical order using Counting Sort.

'''
def counting_sort(characters: List[str]) -> str:
    bucket = {character: 0 for character in ascii_lowercase}
    for character in characters:
        bucket[character] += 1
    
    return ''.join(key * value for key, value in bucket.items() if value > 0)

'''

Given an array Arr[] of N integers. Find the contiguous subarray
(containing at least one number) which has the maximum sum and
return its sum.


'''
def kadanes_algorithm(elements: List[int]) -> int:
    local_max, global_max = elements[0], elements[0]
    for i in range(1, len(elements)):
        # the local max represents the current maximum sum
        # until index i. Once we reach index i, we can choose
        # to expand our chain by including the element at i
        # or start a new chain starting at i. The key thing
        # to note about Kadane's algorithm is that we start a new
        # subarray or chain only when local_max (current_sum)
        # becomes negative
        local_max = max(elements[i], local_max + elements[i])

        # this variable keeps track of the max of the local_max's
        # basically 
        global_max = max(global_max, local_max)

'''

Given an array of integers (possibly some elements negative), write a
program to find out the *maximum product* possible by multiplying 'n'
consecutive integers in the array n <= ARRAY_SIZE. Also, print the
starting point of the maximum product subarray.

'''
def max_product_subarray(elements: List[int]) -> int:
    # case 1: array contains all + elements, in which case the answer
    # is just the product of all elements in the array

    # case 2: array contains a 0, which resets our chain. Therefore
    # the max subarray product up until that point is just arr[0 .. i - 1]
    # in other words, just before that 0. So we start a new chain after 0.

    # case 3: array contains negative numbers. A (-) number will make our
    # product up to that point negative but there is a possibility we will
    # encounter another negative number later on. This is motivation to keep
    # track of the minimum so far as well as the maximum so far because
    # that minimum could potentially turn into a maximum

    local_min = local_max = global_max = elements[0]

    for i in range(1, len(elements)):
        # retain previous local max
        temp = local_max

        # the reason we compute the local max before the local min is because we need the 
        # local_max to be computed based on the previous local_min and the local_min
        # to be computed based on the previous local_max. Therefore, if we computed the
        # local_min first, we would have to first store the previous local_min

        # there are a few possibilities to consider here, let's look at local_max for example
        # the new max, up to index i, can either be the current element itself, an extension
        # of the previous chain (ie continuing the multiplication) or the previous local_min
        # which can be a (-)tive number multipled with the current element (which may also be negative)
        local_max = max(elements[i], local_max * elements[i], local_min * elements[i])
        local_min = min(elements[i], elements[i] * temp, elements[i] * local_min)
        global_max = max(global_max, local_max)
    
    return global_max

