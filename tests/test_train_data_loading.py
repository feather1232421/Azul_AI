import pickle

import numpy as np

from train_mcts_nn import balance_samples_by_player_count, load_and_split_data


def make_sample(marker, player_count=2):
    obs = np.zeros(1108, dtype=np.float32)
    obs[0] = marker
    pi = np.zeros(300, dtype=np.float32)
    pi[0] = 1.0
    z = np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32)
    value_mask = np.zeros(4, dtype=np.float32)
    value_mask[:player_count] = 1.0
    mask = np.zeros(300, dtype=np.float32)
    mask[0] = 1.0
    return obs, pi, z, value_mask, mask


def markers(samples):
    return [int(sample[0][0]) for sample in samples]


def test_mixed_flat_and_episode_data_split_before_training_repeats(tmp_path):
    flat_path = tmp_path / "flat.pkl"
    episode_path = tmp_path / "episodes.pkl"

    with flat_path.open("wb") as f:
        pickle.dump([make_sample(i) for i in range(10)], f)
    with episode_path.open("wb") as f:
        pickle.dump([
            [make_sample(100), make_sample(101)],
            [make_sample(200), make_sample(201)],
        ], f)

    train, val, summaries, paths, split_mode = load_and_split_data(
        data_paths=[flat_path, episode_path],
        repeat_data_paths=[(flat_path, 2)],
        train_ratio=0.5,
        seed=7,
    )

    assert paths == [flat_path, episode_path]
    assert split_mode == "mixed"
    assert len(train) == 17
    assert len(val) == 7
    assert summaries[0]["repeat_count"] == 3
    assert summaries[0]["weighted_train_samples"] == 15

    train_markers = markers(train)
    val_markers = set(markers(val))
    assert set(train_markers).isdisjoint(val_markers)

    repeated_flat_markers = [marker for marker in train_markers if marker < 100]
    assert len(repeated_flat_markers) == 15
    assert all(repeated_flat_markers.count(marker) == 3 for marker in set(repeated_flat_markers))


def test_balance_samples_by_player_count_uses_exact_2_1_1_sample_mix():
    samples = (
        [make_sample(i, 2) for i in range(40)]
        + [make_sample(100 + i, 3) for i in range(10)]
        + [make_sample(200 + i, 4) for i in range(12)]
    )

    balanced, summary = balance_samples_by_player_count(
        samples,
        {2: 2, 3: 1, 4: 1},
        seed=7,
    )

    assert len(balanced) == 40
    assert summary["available"] == {2: 40, 3: 10, 4: 12}
    assert summary["selected"] == {2: 20, 3: 10, 4: 10}
