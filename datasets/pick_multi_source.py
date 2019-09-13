#!/usr/bin/env python3
"""
Generates the list of which multi-source adaptation problems to perform

For each dataset, for each target user, pick n random source users (excluding
the target user) 3 different times (so we can get mean +/- stdev).
"""
import random


def zero_to_n(n):
    """ Return [0, 1, 2, ..., n] """
    return list(range(0, n+1))


def one_to_n(n):
    """ Return [1, 2, 3, ..., n] """
    return list(range(1, n+1))


def other_users(users, skip_user):
    """ From the list of users, throw out skip_user """
    new_users = []

    for user in users:
        if user != skip_user:
            new_users.append(user)

    return new_users


def generate_n_with_max(num_users, max_num):
    """ Generate [1,2,3,...,num_users] but max out at max_num and skip as close
    to evenly to get there. For example, if num_users=30 and max_num=5, we get:
    [1, 7, 13, 19, 25].
    """
    return list(range(1, num_users, num_users//max_num))[:max_num]


def generate_multi_source(dataset_name, users, n, repeat=3, max_users=5):
    # Shrink the number of target users since otherwise we have >4000 adaptation
    # problems. That will take too long and won't fit in the paper's table
    # anyway.
    possible_target_users = users[:max_users]

    # Output strings - ignore duplicates in datasets.py by indexing by name
    for_tfrecords = []
    for_datasets = {}
    pairs = []

    # We'll generate multi-source options for each target user
    for target_user in possible_target_users:
        already_used_target = {}

        # We want several random subsets of each so we can get mean +/- stdev
        for i in range(repeat):
            # Select random source domains excluding target, keep shuffling until
            # we find a source set we haven't already used. The point of "repeat"
            # is to get *different* subsets. If it's the same, then there's not
            # much point in re-running with the exact same data.
            j = 0
            while True:
                others = other_users(users, target_user)
                random.shuffle(others)
                assert n <= len(others), "cannot choose n larger than len(users)-1"
                source_users = others[:n]

                if tuple(source_users) not in already_used_target:
                    already_used_target[tuple(source_users)] = None
                    break
                elif j > 1000:
                    print("Infinite loop...")
                    exit(1)
                j += 1

            source_users = [str(x) for x in source_users]

            source = "\"" + dataset_name + "_n" + str(n) + "_" + ",".join(source_users) + "\""
            target = "\"" + dataset_name + "_t" + str(target_user) + "\""

            for_tfrecords.append("(" + source + ", " + target + "),")
            for_datasets[source] = source + ": make_" + dataset_name \
                + "(users=[" + ",".join(source_users) + "]),"
            for_datasets[target] = target + ": make_" + dataset_name \
                + "(users=[" + str(target_user) + "], target=True),"
            pairs.append((source, target))

    # We only used the dictionary for removing duplicates, now convert
    # back to a list
    for_datasets = list(for_datasets.values())

    return for_datasets, for_tfrecords, pairs


if __name__ == "__main__":
    # List of datasets and users in each
    datasets = {
        "ucihar": one_to_n(30),
        "uwave": one_to_n(8),
        "sleep": zero_to_n(25),
    }

    # Output strings
    for_tfrecords = []
    for_datasets = []
    pairs = []

    for name, users in datasets.items():
        # For each value of n, from 1 (single-source domain adaptation) up to
        # the full number of users - 1 (since we have one for the target)
        for n in generate_n_with_max(len(users), 5):
            # Make this repeatable even if we change which datasets, how many
            # n's we use, etc. Also nice since we end up using a subset of
            # n's source domains as (n-1)'s source domains. For example,
            # we get
            # "sleep_17", "sleep_t0"
            # "sleep_17,13", "sleep_t0"
            # "sleep_17,13,10", "sleep_t0"
            # "sleep_17,13,10,20", "sleep_t0"
            random.seed(42)

            curr_datasets, curr_tfrecords, curr_pairs = generate_multi_source(name, users, n)
            for_datasets += curr_datasets
            for_tfrecords += curr_tfrecords
            pairs += curr_pairs

    # Remove dataset duplicates (still could be duplicates due to target)
    for_datasets = list(set(for_datasets))

    # Sort
    for_datasets.sort()
    for_tfrecords.sort()

    # Print
    print("For generate_tfrecords.py:")
    for r in for_tfrecords:
        print(r)
    print()

    print("For datasets.py:")
    for r in for_datasets:
        print(r)
    print()

    print("For kamiak_{train,eval}_real.srun:")
    sources = []
    targets = []
    for source, target in pairs:
        sources.append(source)
        targets.append(target)

    print("# number of adaptation problems =", len(sources))
    print("sources=(", " ".join(sources), ")", sep="")
    print("targets=(", " ".join(targets), ")", sep="")
