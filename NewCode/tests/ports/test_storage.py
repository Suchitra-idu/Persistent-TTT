from __future__ import annotations

import pytest

from ttt.adapters.in_memory_storage import InMemoryStorage
from ttt.adapters.local_storage import LocalStorage
from ttt.ports.storage import Storage

PATH = "run/ckpt-100/ttt_params.pt"
DATA = b"\x00checkpoint\xff"


class StorageConformance:
    @pytest.fixture
    def storage(self):
        raise NotImplementedError

    def test_it_satisfies_the_port(self, storage):
        assert isinstance(storage, Storage)

    def test_an_absent_path_does_not_exist(self, storage):
        assert not storage.exists(PATH)

    def test_a_written_path_exists(self, storage):
        storage.write_bytes(PATH, DATA)

        assert storage.exists(PATH)

    def test_bytes_round_trip(self, storage):
        storage.write_bytes(PATH, DATA)

        assert storage.read_bytes(PATH) == DATA

    def test_reading_an_absent_path_raises(self, storage):
        with pytest.raises(FileNotFoundError):
            storage.read_bytes(PATH)

    def test_a_second_write_overwrites(self, storage):
        storage.write_bytes(PATH, DATA)
        storage.write_bytes(PATH, b"newer")

        assert storage.read_bytes(PATH) == b"newer"

    def test_listing_finds_what_was_written_under_the_prefix(self, storage):
        storage.write_bytes(PATH, DATA)
        storage.write_bytes("run/ckpt-200/ttt_params.pt", DATA)

        assert storage.listdir("run") == (PATH, "run/ckpt-200/ttt_params.pt")

    def test_listing_excludes_other_prefixes(self, storage):
        storage.write_bytes(PATH, DATA)
        storage.write_bytes("other/x.pt", DATA)

        assert storage.listdir("run") == (PATH,)

    def test_listing_an_absent_prefix_is_empty(self, storage):
        assert storage.listdir("nothing/here") == ()

    def test_commit_is_safe_to_call(self, storage):
        storage.write_bytes(PATH, DATA)
        storage.commit()

        assert storage.read_bytes(PATH) == DATA


class TestInMemoryStorage(StorageConformance):
    @pytest.fixture
    def storage(self):
        return InMemoryStorage()

    def test_commits_are_counted(self, storage):
        storage.commit()
        storage.commit()

        assert storage.commits == 2


class TestLocalStorage(StorageConformance):
    @pytest.fixture
    def storage(self, tmp_path):
        return LocalStorage(tmp_path)

    def test_writing_creates_intermediate_directories(self, storage, tmp_path):
        storage.write_bytes(PATH, DATA)

        assert (tmp_path / PATH).is_file()

    def test_a_path_escaping_the_root_is_rejected(self, storage):
        with pytest.raises(ValueError, match="escapes the storage root"):
            storage.write_bytes("../outside.pt", DATA)
