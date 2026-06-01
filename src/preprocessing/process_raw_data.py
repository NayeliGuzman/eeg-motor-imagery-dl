
import os
import sys
import numpy as np
import mne
from sklearn.model_selection import train_test_split

RANDOM_STATE = 12345
TEST_SIZE = 50

def process_subject(filepath, subject_id):
    """Load, filter, and epoch a single subject's .gdf file."""
    raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)
    raw.drop_channels(['EOG-left', 'EOG-central', 'EOG-right'])
    raw.filter(l_freq=4.0, h_freq=40.0, verbose=False)

    events, _ = mne.events_from_annotations(raw, verbose=False)

    epochs = mne.Epochs(
        raw, events,
        event_id={'left_hand': 7, 'right_hand': 8, 'feet': 9, 'tongue': 10},
        tmin=0.5, tmax=4.496,
        baseline=None, preload=True,
        reject_by_annotation=True,
        verbose=False
    )

    X = epochs.get_data()           # (n_trials, 22, 1000)
    y = epochs.events[:, 2] - 7    # remap 7-10 to 0-3

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    person_train = np.full((len(y_train), 1), subject_id)
    person_test  = np.full((len(y_test),  1), subject_id)

    print(f"  Subject {subject_id}: train={len(y_train)}, test={len(y_test)}")

    return X_train, X_test, y_train, y_test, person_train, person_test


def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    all_X_train, all_y_train, all_person_train = [], [], []
    all_X_test,  all_y_test,  all_person_test  = [], [], []

    # Get all T files, exclude A04T and all E files
    gdf_files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith('T.gdf') and f != 'A04T.gdf'
    ])

    print(f"Found {len(gdf_files)} subject files: {gdf_files}")

    for filename in gdf_files:
        # Extract subject number from filename e.g. A01T.gdf -> 0
        subject_id = int(filename[1:3]) - 1
        filepath   = os.path.join(input_dir, filename)

        print(f"Processing {filename}...")
        X_train, X_test, y_train, y_test, person_train, person_test = process_subject(
            filepath, subject_id
        )

        all_X_train.append(X_train)
        all_y_train.append(y_train)
        all_person_train.append(person_train)
        all_X_test.append(X_test)
        all_y_test.append(y_test)
        all_person_test.append(person_test)

    # Concatenate
    X_train_valid     = np.concatenate(all_X_train, axis=0)
    y_train_valid     = np.concatenate(all_y_train, axis=0)
    person_train_valid = np.concatenate(all_person_train, axis=0)
    X_test            = np.concatenate(all_X_test,  axis=0)
    y_test            = np.concatenate(all_y_test,  axis=0)
    person_test       = np.concatenate(all_person_test,  axis=0)

    # Save
    np.save(os.path.join(output_dir, 'X_train_valid.npy'),      X_train_valid)
    np.save(os.path.join(output_dir, 'y_train_valid.npy'),      y_train_valid)
    np.save(os.path.join(output_dir, 'person_train_valid.npy'), person_train_valid)
    np.save(os.path.join(output_dir, 'X_test.npy'),             X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'),             y_test)
    np.save(os.path.join(output_dir, 'person_test.npy'),        person_test)

    print(f"\nX_train_valid shape : {X_train_valid.shape}")
    print(f"y_train_valid shape : {y_train_valid.shape}")
    print(f"X_test shape        : {X_test.shape}")
    print(f"y_test shape        : {y_test.shape}")
    print(f"\nSaved all .npy files to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python process_raw_data.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir  = sys.argv[1]
    output_dir = sys.argv[2]
    main(input_dir, output_dir)