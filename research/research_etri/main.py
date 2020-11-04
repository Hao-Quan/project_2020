# import time
# import multiprocessing
#
#
# def is_prime(n):
#     if (n <= 1):
#         return 'not a prime number'
#     if (n <= 3):
#         return 'prime number'
#
#     if (n % 2 == 0 or n % 3 == 0):
#         return 'not a prime number'
#
#     i = 5
#     while (i * i <= n):
#         if (n % i == 0 or n % (i + 2) == 0):
#             return 'not a prime number'
#         i = i + 6
#
#     return 'prime number'
#
#
# def multiprocessing_func(x):
#     time.sleep(2)
#     print('{} is {} number'.format(x, is_prime(x)))
#
#
# if __name__ == '__main__':
#     starttime = time.time()
#     processes = []
#     for i in range(1, 10):
#         p = multiprocessing.Process(target=multiprocessing_func, args=(i,))
#         processes.append(p)
#         p.start()
#
#     for process in processes:
#         process.join()
#
#     print()
#     print('Time taken = {} seconds'.format(time.time() - starttime))


# from multiprocessing import Pool
# import time
#
# def f(x):
#     return x*x
#
# if __name__ == '__main__':
#     with Pool(processes=4) as pool:         # start 4 worker processes
#         result = pool.apply_async(f, (10,)) # evaluate "f(10)" asynchronously in a single process
#         print(result.get(timeout=1))        # prints "100" unless your computer is *very* slow
#
#         print(pool.map(f, range(10)))       # prints "[0, 1, 4,..., 81]"
#
#         it = pool.imap(f, range(10))
#         print(next(it))                     # prints "0"
#         print(next(it))                     # prints "1"
#         print(it.next(timeout=1))           # prints "4" unless your computer is *very* slow
#
#         result = pool.apply_async(time.sleep, (10,))
#         print(result.get(timeout=1))


from multiprocessing import Pool
import time

def worker(x, y):
    """worker function"""
    result = x * y
    return result

if __name__ == "__main__":
    # p = Pool(processes=20)
    # data = p.starmap(worker, ([1,2], [3,4]))
    # p.close()

    start_time = time.time()
    p = Pool(processes=20)
    data = p.starmap(worker, ([1,2], [3,4]))
    p.close()
    print("--- %s seconds ---" % (time.time() - start_time)/60)
