import numpy as np
import pytest
import tensorflow as tf

from tensorversehub import data_utils as du
from tensorversehub.compat import keras


class TestTFRecordHandler:
    @pytest.mark.parametrize(
        "value, kind",
        [
            (3, "int64_list"),
            (True, "int64_list"),
            (np.int32(4), "int64_list"),
            (1.5, "float_list"),
            (np.float32(2.5), "float_list"),
            ("text", "bytes_list"),
            (b"raw", "bytes_list"),
            ([1, 2, 3], "int64_list"),
            ([0.1, 0.2], "float_list"),
            (["a", "b"], "bytes_list"),
            (np.zeros((2, 2), np.float32), "float_list"),
            (tf.constant([1, 2]), "int64_list"),
        ],
    )
    def test_to_feature_dispatch(self, value, kind):
        feature = du.TFRecordHandler.to_feature(value)
        assert feature.HasField(kind)

    def test_to_feature_rejects_unknown(self):
        with pytest.raises(TypeError):
            du.TFRecordHandler.to_feature(object())

    def test_text_roundtrip(self, tmp_path):
        handler = du.TFRecordHandler()
        examples = [
            handler.serialize_text_example(f"hello world {i}", i % 2, {"score": 0.5})
            for i in range(6)
        ]
        path = tmp_path / "text.tfrecord"
        assert handler.write_tfrecord(examples, path, compression="GZIP") == 6
        assert du.TFRecordHandler.count_records(path, compression="GZIP") == 6
        ds = tf.data.TFRecordDataset(str(path), compression_type="GZIP").map(du.parse_text_tfrecord)
        text, label = next(iter(ds))
        assert text.numpy().startswith(b"hello world") and int(label) == 0

    def test_array_roundtrip_sharded(self, tmp_path):
        handler = du.TFRecordHandler()
        arrays = [np.random.rand(4, 4, 3).astype("float32") for _ in range(7)]
        examples = [
            handler.serialize_array_example(a, i, {"aux": [1, 2]}) for i, a in enumerate(arrays)
        ]
        paths = handler.write_sharded(examples, tmp_path / "shards", prefix="arr", num_shards=3)
        assert len(paths) == 3 and len(du.expand_paths(tmp_path / "shards")) == 3
        assert len(du.expand_paths(str(tmp_path / "shards" / "arr-*.tfrecord"))) == 3
        ds = du.create_tfrecord_dataset(
            paths,
            batch_size=7,
            shuffle_buffer=0,
            parse_fn=lambda p: du.parse_array_tfrecord(p, tf.float32),
        )
        x, y = next(iter(ds))
        assert x.shape == (7, 4, 4, 3)
        assert sorted(y.numpy().tolist()) == list(range(7))
        with pytest.raises(ValueError):
            handler.write_sharded(examples, tmp_path, num_shards=0)

    def test_image_roundtrip(self, tmp_path):
        img = (np.random.rand(8, 6, 3) * 255).astype("uint8")
        png = tmp_path / "img.png"
        tf.io.write_file(str(png), tf.io.encode_png(img))
        handler = du.TFRecordHandler()
        example = handler.serialize_image_example(png, 2, {"source": "unit-test"})
        parsed = tf.io.parse_single_example(example, du.create_feature_description_image())
        assert int(parsed["height"]) == 8 and int(parsed["width"]) == 6
        image, label = du.parse_image_tfrecord(tf.constant(example), image_size=(4, 4))
        assert image.shape == (4, 4, 3) and int(label) == 2 and float(tf.reduce_max(image)) <= 1.0

    def test_missing_files(self):
        with pytest.raises(FileNotFoundError):
            du.DataPipeline().create_tfrecord_dataset(
                ["/nonexistent/*.tfrecord"], parse_fn=lambda p: p
            )
        with pytest.raises(ValueError):
            du.DataPipeline().create_tfrecord_dataset(["x.tfrecord"])


class TestImageOps:
    def test_rotate_identity_and_shapes(self):
        img = tf.random.uniform((12, 10, 3))
        out = du.rotate_image(img, 0.0)
        assert out.shape == img.shape
        np.testing.assert_allclose(out.numpy(), img.numpy(), atol=1e-5)
        batch = du.rotate_image(tf.random.uniform((2, 12, 10, 1)), 0.5)
        assert batch.shape == (2, 12, 10, 1)
        rotated = du.rotate_image(img, 1.0)
        assert not np.allclose(rotated.numpy(), img.numpy())
        assert du.rotate_image(tf.cast(img * 255, tf.uint8), 0.2).dtype == tf.uint8

    def test_augment_image_range(self):
        img = tf.random.uniform((16, 16, 3))
        out = du.augment_image(img)
        assert out.shape == (16, 16, 3)
        assert float(tf.reduce_min(out)) >= 0.0 and float(tf.reduce_max(out)) <= 1.0
        gray = du.augment_image(tf.random.uniform((16, 16, 1)))
        assert gray.shape == (16, 16, 1)


class TestDataPipeline:
    def test_invalid_batch_size(self):
        with pytest.raises(ValueError):
            du.DataPipeline(batch_size=0)

    def test_from_arrays(self, images, labels):
        pipe = du.DataPipeline(batch_size=8, shuffle_buffer=10, drop_remainder=True)
        ds = pipe.from_arrays(images, labels, map_fn=lambda x, y: (x * 2, y))
        x, y = next(iter(ds))
        assert x.shape == (8, 16, 16, 3) and y.shape == (8,)
        assert float(tf.reduce_max(x)) <= 2.0
        assert int(ds.cardinality()) == 3

    def test_image_dataset_from_files(self, tmp_path):
        paths = []
        for i in range(6):
            img = (np.random.rand(10, 10, 3) * 255).astype("uint8")
            p = tmp_path / f"{i}.png"
            tf.io.write_file(str(p), tf.io.encode_png(img))
            paths.append(str(p))
        pipe = du.DataPipeline(batch_size=3, shuffle_buffer=6)
        ds = pipe.create_image_dataset(paths, [0, 1, 0, 1, 0, 1], image_size=(8, 8), augment=True)
        x, y = next(iter(ds))
        assert x.shape == (3, 8, 8, 3) and x.dtype == tf.float32
        with pytest.raises(ValueError):
            pipe.create_image_dataset(paths, [0])

    def test_text_dataset(self):
        pipe = du.DataPipeline(batch_size=4, shuffle_buffer=0)
        texts = ["the cat sat", "a dog ran fast", "hello"] * 4
        ds, vectorizer = pipe.create_text_dataset(texts, [0, 1, 2] * 4, max_length=6, vocab_size=50)
        x, y = next(iter(ds))
        assert x.shape == (4, 6) and isinstance(vectorizer, keras.layers.TextVectorization)
        ds2, same = pipe.create_text_dataset(texts, [0] * 12, max_length=6, vectorizer=vectorizer)
        assert same is vectorizer

    def test_mixed_precision_dataset(self, image_dataset):
        x, _ = next(iter(du.DataPipeline.create_mixed_precision_dataset(image_dataset)))
        assert x.dtype == tf.float16
        dict_ds = image_dataset.map(lambda x, y: ({"a": x, "b": tf.cast(y, tf.int32)}, y))
        feats, _ = next(iter(du.DataPipeline.create_mixed_precision_dataset(dict_ds)))
        assert feats["a"].dtype == tf.float16 and feats["b"].dtype == tf.int32


class TestAugmentation:
    def test_augmentation_layer(self, images):
        layer = du.DataAugmentation.create_augmentation_layer(seed=1)
        out = layer(images, training=True)
        assert out.shape == images.shape

    def test_mixup_and_cutmix(self, images, labels):
        for fn in (du.DataAugmentation.mixup_batch, du.DataAugmentation.cutmix_batch):
            x, y = fn(images, labels, num_classes=3)
            assert x.shape == images.shape and y.shape == (len(labels), 3)
            np.testing.assert_allclose(tf.reduce_sum(y, axis=1).numpy(), 1.0, atol=1e-5)
            assert float(tf.reduce_min(x)) >= 0.0 and float(tf.reduce_max(x)) <= 1.0
        with pytest.raises(ValueError):
            du.DataAugmentation.mixup_batch(images, labels)  # sparse labels need num_classes
        one_hot = tf.one_hot(labels, 3)
        _, y = du.DataAugmentation.cutmix_batch(images, one_hot)
        assert y.shape == (len(labels), 3)

    def test_dataset_wrappers(self, image_dataset):
        for wrapped in (
            du.DataAugmentation.mixup(image_dataset, num_classes=3),
            du.DataAugmentation.cutmix(image_dataset, num_classes=3),
        ):
            x, y = next(iter(wrapped))
            assert x.shape[1:] == (16, 16, 3) and y.shape[1] == 3

    def test_random_erasing(self):
        img = tf.ones((16, 16, 3))
        erased = du.DataAugmentation.random_erasing(img, probability=1.0)
        assert float(tf.reduce_min(erased)) == 0.0 and erased.shape == img.shape
        same = du.DataAugmentation.random_erasing(img, probability=0.0)
        np.testing.assert_array_equal(same.numpy(), img.numpy())


class TestConvenience:
    def test_image_classification_pipeline(self, tmp_path):
        for cls in ("cat", "dog"):
            (tmp_path / cls).mkdir()
            for i in range(5):
                img = (np.random.rand(12, 12, 3) * 255).astype("uint8")
                tf.io.write_file(str(tmp_path / cls / f"{i}.png"), tf.io.encode_png(img))
        train_ds, val_ds = du.create_image_classification_pipeline(
            tmp_path, batch_size=4, image_size=(8, 8), validation_split=0.2, cache=False
        )
        x, y = next(iter(train_ds))
        assert x.shape[1:] == (8, 8, 3) and float(tf.reduce_max(x)) <= 1.0
        assert int(val_ds.cardinality()) >= 1

    def test_text_classification_pipeline(self):
        texts = [f"sample text number {i}" for i in range(20)]
        labels = [i % 2 for i in range(20)]
        train, val, vec = du.create_text_classification_pipeline(
            texts, labels, batch_size=4, sequence_length=5
        )
        x, y = next(iter(train))
        assert x.shape == (4, 5)
        assert sum(int(b[0].shape[0]) for b in val) == 4
        with pytest.raises(ValueError):
            du.create_text_classification_pipeline(texts, labels[:-1])

    def test_split_dataset_and_class_weights(self):
        ds = tf.data.Dataset.range(10)
        a, b, c = du.split_dataset(ds, (0.5, 0.3, 0.2))
        assert [int(s.cardinality()) for s in (a, b, c)] == [5, 3, 2]
        with pytest.raises(ValueError):
            du.split_dataset(ds, (0.5, 0.6))
        with pytest.raises(ValueError):
            du.split_dataset(ds.repeat(), (0.5, 0.5))
        weights = du.compute_class_weights([0, 0, 0, 1])
        assert weights[1] > weights[0] and pytest.approx(weights[0]) == 4 / (2 * 3)
