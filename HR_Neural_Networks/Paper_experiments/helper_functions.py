import numpy as np

def return_batches_to_corrupt(iter, train_batches, train_batch_size, frac, mis_specification, num_classes):
    np.random.seed(iter)
    sampled_batches = np.random.choice(
        np.arange(0, train_batches), replace=False, size=int(frac*train_batches))

    # Misspecification - corrupting labels at random
    batch_starts = [batch_no * train_batch_size for batch_no in sampled_batches]
    batch_ends = [min(bstart + train_batch_size, train_batches *
                      train_batch_size) for bstart in batch_starts]
    data_points_in_batches = np.array([np.arange(
        batch_starts[i], batch_ends[i]) for i in range(len(batch_starts))]).flatten()
    number_to_misspecify = int(len(data_points_in_batches)*mis_specification)
    data_points_to_corrupt = np.random.choice(
        data_points_in_batches, size=number_to_misspecify, replace=False)
    unique_labels = np.arange(0, num_classes)
    return sampled_batches, data_points_to_corrupt, unique_labels

def corrupt_targets(batch_idx, 
                    targets, 
                    sampled_batches, 
                    train_batch_size,
                    train_batches,
                    data_points_to_corrupt,
                    unique_labels):

  if batch_idx in sampled_batches:

    batch_start = batch_idx * train_batch_size
    batch_end = min(batch_start + train_batch_size, train_batches*train_batch_size)

    to_corrupt = []
    idx_to_corrupt = 0

    for i in range(batch_start, batch_end):
      
      if i in data_points_to_corrupt:
        to_corrupt.append(idx_to_corrupt)

      idx_to_corrupt += 1

    orig_y_train = targets.clone()

    for data_point in to_corrupt:

        random_label = np.random.choice(np.setdiff1d(
                                        unique_labels, 
                                        orig_y_train[data_point]))

        targets[data_point] = random_label

  return targets


