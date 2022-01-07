# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 23:07:54 2021

@author: Tashfeen Mustafa Choudhury

Program: Apriori Algorithm for finding frequent itemsets
"""

import sys
import time
import numpy as np


def load_dataset(dataset_name):
    path = 'E:\IUB-Courses\Autumn 2021\Data Mining & Warehouse\Assignment 3 - FrequentItemset\AprioriAlgorithm\data'
    input_file_path = path + '\\' + dataset_name
    data = None
    try:
        data = np.loadtxt(input_file_path, dtype='int64')
    except:
        # if baskets are uneven
        # read data like this
        data = []
        file = open(input_file_path)

        for line in file:
            line = line.strip().split(' ')
            temp = []
            for l in line:
                temp.append(int(l))
            data.append(temp)

    return data


def print_summary(data):
    average_transaction_length = 0

    for d in data:
        average_transaction_length += len(d)

    average_transaction_length /= len(data)

    print('Total Number of Transactions: ', len(data))
    print('Average Transaction Length: ', average_transaction_length)


def find_frequent_l_items(data, min_sup):
    itemset = None

    # create first candidate list (c_1)
    c_1 = []

    # loops through data and updates l_1 with itemset
    for d in data:
        for j in range(len(d)):
            # print('j: ', j)
            itemset = d[j]

            # assuming itemset is new item
            new_item = True

            c_1_length = len(c_1)

            if c_1_length > 0:

                # check if itemset exists
                for i in range(c_1_length):
                    # print('itemset: ', itemset)
                    # print('c_1[i][0]: ', c_1[i][0])
                    # print('i: ', i)
                    if itemset == c_1[i][0]:
                        # if it does, increment itemset count
                        # print('here')
                        c_1[i][1] += 1
                        new_item = False

            # if itemset is new item append to c_1
            if new_item:
                c_1.append([itemset, 1])
            # print(c_1)

    # create l_1 frequent itemsets
    l_1 = []

    for c in c_1:
        # only add the itemsets that have count >= min_sup
        if c[1] >= min_sup:
            l_1.append(c)

    # sort the list
    l_1.sort()

    return l_1


def has_infrequent_subset(c, l_k):
    is_infrequent = True

    # For each c_ in c check if it is in l_k
    for index_c_, c_ in enumerate(c):
        is_infrequent = True

        for index_l, l in enumerate(l_k):

            # if it is in l_k then it is not infrequent
            for l_ in l[:-1]:
                if l_ == c_:
                    is_infrequent = False
                    break

    return is_infrequent


def apriori_generator(l_k, k):
    c_k = []

    # execute if k == 2 (i.e. 1-itemset)
    if k == 2:
        # loop through l_k twice
        for i in range(len(l_k)):
            for j in range(i + 1, len(l_k)):

                # concatenate itemsets
                c = l_k[i][:-1] + l_k[j][:-1]

                # check if they are infrequents subsets
                if not has_infrequent_subset(c, l_k):
                    # if not, add them to c_k
                    c_k.append(c)

    # execute if k > 2 (i.e greater than 1-itemset)
    else:
        # initialize join criteria to False
        fulfill_join_criteria = False

        # loop l_k twice
        for i in range(len(l_k) - 1):
            # grab (k - 1)-itemset for l_i (first itemset)
            l_i = l_k[i][:k - 1]

            # initialize second (k - 1)-itemset
            l_j = None
            for j in range(i + 1, len(l_k)):
                # grab (k - 1)-itemset for l_i (first itemset)
                l_j = l_k[j][:k - 1]

                # if all elements of both (k - 1)-itemsets are same and last element
                # of l_i is smaller than l_j than fulfill join criteria is True and break
                # else continue to the next l_j
                if l_i[:-1] == l_j[:-1] and l_i[-1] < l_j[-1]:
                    fulfill_join_criteria = True
                    break
                else:
                    continue

            # if join criteria is fulfilled
            if fulfill_join_criteria:
                # initialize temp_c
                temp_c = []

                # loop through l_i
                for i in range(len(l_i)):
                    # if a value of l_i and l_j are not already in temp_c
                    # then append them to temp_c
                    if l_i[i] not in temp_c:
                        temp_c.append(l_i[i])
                    if l_j[i] not in temp_c:
                        temp_c.append(l_j[i])

                # sort temp_c smallest to largest
                temp_c.sort()

                # append temp_c to c_k
                c_k.append(temp_c)

    return c_k


def apriori(data, min_sup, dataset_name):
    l_k = find_frequent_l_items(data, min_sup)

    print('l_k: ', l_k)
    print('\n')

    output_txt(l_k, dataset_name, k=1)

    k = 2

    while len(l_k) > 0:
        c_k = apriori_generator(l_k, k)
        print('c_k: ', c_k)
        print('\n')
        # next candidate list initialization
        temp_c_k = []

        # to check if itemset exists in a basket
        is_in_d = False

        # check each basket
        for d in data:
            # check each itemset in c_k
            for c_t in c_k:
                # check if each item in itemset is in d(basket)
                for c in c_t:
                    is_in_d = False
                    if c in d:
                        is_in_d = True

                # if itemset is in basket
                if is_in_d:
                    # assuming itemset is new item
                    new_item = True

                    # add itemset to temp_c_k
                    if len(temp_c_k) > 0:
                        for i in range(len(temp_c_k)):
                            # if itemset already exists in temp_c_k
                            if temp_c_k[i][:-1] == c_t:
                                # update count of itemset
                                temp_c_k[i][-1] += 1
                                new_item = False
                    # if new item append to itemset        
                    if new_item:
                        temp_c_k.append(c_t + [1])

        print('temp_c_k: ', temp_c_k)
        print('\n')
        # now add candidate itemsets from temp_c_k to l_k where count for
        # each candidate >= min_sup
        new_l_k = []

        for temp_c in temp_c_k:
            # only add the itemsets that have count >= min_sup
            if temp_c[-1] >= min_sup:
                new_l_k.append(temp_c)

        # sort the list
        new_l_k.sort()

        # update l_k with new_l_k
        l_k = new_l_k

        print('new_l_k: ', l_k)
        print('\n')
        output_txt(l_k, dataset_name, k)

        k += 1

    return


def output_txt(l_k, dataset_name, k):
    file_name = 'result_' + dataset_name[:-4] + '.txt'
    mode = ''
    if k == 1:
        mode = 'w'
    else:
        mode = 'a'

    file = open(file_name, mode)

    file.write('L-' + str(k) + '\n' + str(l_k) + '\n')

    file.close()


def main():
    # start time
    start = time.time()

    # taking minimum support from sys.argv
    min_sup = int(sys.argv[1])
    # taking dataset name from sys.argv
    dataset_name = sys.argv[2]

    # load the dataset
    data = load_dataset(dataset_name)

    print_summary(data)

    # Perform Apriori Algorithm
    apriori(data, min_sup, dataset_name)

    # end time
    print('Total Elapsed Time: %s' % (time.time() - start))


if __name__ == "__main__":
    main()
