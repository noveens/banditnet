import numpy as np
import pickle
from tqdm import tqdm

avg_num_zeros = 0.0

def one_hot(arr):
    new = []
    for i in tqdm(range(len(arr))):
        temp = np.zeros(10)
        temp[arr[i]] = 1.0
        new.append(temp)
    return np.array(new)

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_probs(label):
    probs = np.power(np.add(list(range(10)), 1), 0.5)
    probs[label] = probs[label] * 10

    probs = probs / probs.sum()

    return probs

x_train, x_val, x_test = [], [], []
y_train, y_val, y_test = [], [], []

# Train
for b in range(1, 6):
    this_batch = unpickle("data/cifar-10-batches-py/data_batch_" + str(b))

    if len(x_train) == 0: x_train, y_train = this_batch[b'data'], this_batch[b'labels']
    else: 
        x_train = np.concatenate((x_train, this_batch[b'data']), axis=0)
        y_train = np.concatenate((y_train, this_batch[b'labels']), axis=0)

# Test
this_batch = unpickle("data/cifar-10-batches-py/test_batch")
x_test, y_test = this_batch[b'data'], this_batch[b'labels']

# Normalize X data
x_train = x_train.astype(float) / 255.0
x_test = x_test.astype(float) / 255.0

# One hot the rewards
y_train = one_hot(y_train)
y_test = one_hot(y_test)

# Shuffle the dataset once
indices = np.arange(len(x_train))
np.random.shuffle(indices)
assert len(x_train) == len(y_train)
x_train = x_train[indices]
y_train = y_train[indices]

# Start creating bandit-dataset
for num_sample in [1]: # [1, 2, 3, 4, 5]:
            
    print("Pre-processing for num sample = " + str(num_sample))

    final_x, final_y, final_actions, final_prop = [], [], [], []

    avg_num_zeros = 0.0
    expected_reward = 0.0
    total = 0.0

    for epoch in range(num_sample):
        for point_num in tqdm(range(x_train.shape[0])):
            
            label = np.argmax(y_train[point_num])
            image = x_train[point_num]

            probs = get_probs(label)

            actionvec = np.random.multinomial(1, probs)
            action = np.argmax(actionvec)
            
            expected_reward += float(int(action == label))
            total += 1.0

            # Printing the first prob. dist.
            # if point_num == 0: print("Prob Distr. for 0th sample:\n", [ round(i, 3) for i in list(probs) ])

            final_x.append(image)
            final_y.append(y_train[point_num])
            final_actions.append([ action ])
            final_prop.append(probs)

    avg_num_zeros /= float(x_train.shape[0])
    avg_num_zeros = round(avg_num_zeros, 4)
    print("Num sample = " + str(num_sample) + "; Acc = " + str(100.0 * expected_reward / total))
    print()

    # Save as CSV
    final_normal = np.concatenate((final_x, final_y, final_prop, final_actions), axis=1)

    train = final_normal
    val = [] # No validation set rn

    test_prop, test_actions = [], []
    for label in y_test:
        probs = get_probs(np.argmax(label))
        test_prop.append(probs)
        actionvec = np.random.multinomial(1, probs)
        action = np.argmax(actionvec)
        test_actions.append([ action ])
    test = np.concatenate((x_test, y_test, test_prop, test_actions), axis=1) # Garbage

    filename  = 'data/cifar-10-batches-py/bandit_data_'
    filename += 'sampled_' + str(num_sample)

    save_obj(train, filename + '_train')
    save_obj(test, filename + '_test')
    save_obj(val, filename + '_val') # No validation set right now, directly evaluating on test
