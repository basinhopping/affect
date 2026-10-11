"""The standard utterance table and the interface every dataset module implements."""
import dataclasses


@dataclasses.dataclass(frozen=True)
class Utterance:
    """One row of the standard utterance table. Every dataset produces these."""

    utt_id: str
    dataset: str
    speaker: str
    fold_group: str        # the unit held out by the fold scheme (IEMOCAP: the session, "S1" to "S5")
    dialog: str
    kind: str              # IEMOCAP: "impro" or "script"
    raw_label: str         # the label as the dataset publishes it
    label: str             # the label after the chosen label set's mapping
    audio_path: str        # relative to the dataset's assembled folder, with forward slashes
    n_frames: int
    sample_rate: int
    transcript: str | None
    extra: dict            # dataset-specific fields, e.g. IEMOCAP's valence, activation, votes

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.sample_rate


class Dataset:
    """Interface of a dataset module. Subclasses are registered with @register_dataset."""

    name = ""
    sample_rate = 0                   # every file's sample rate; checked when the data is assembled
    assembly_sources: tuple = ()      # names of the folders an assembly reads, given on the command line
    manifests_required: tuple = ()    # raw folders that must be ingested first (their manifests are used)
    meta_files: frozenset = frozenset()  # files in assembled/ that are not data (manifest, record)

    def assembly_listings(self, sources: dict) -> dict:
        """Name -> folder, for every folder the assembly plan needs listed."""
        raise NotImplementedError

    def plan_assembly(self, sources: dict, listings: dict, manifests: dict) -> tuple:
        """(copy items, report of the checks that need only names and sizes)."""
        raise NotImplementedError

    def check_after_copy(self, rows: list, manifests: dict) -> dict:
        """Checks that need the hashes made while copying. `ok` True only if all passed."""
        raise NotImplementedError

    def check_assembled(self, root, manifest_rows: list) -> dict:
        """Checks of an assembled copy. The report's `ok` is True only if all passed."""
        raise NotImplementedError

    def class_order(self, settings) -> tuple[str, ...]:
        """Class names in output-unit order for the settings' label set."""
        raise NotImplementedError

    def utterances(self, settings) -> list:
        """The standard utterance table for these settings, sorted by utt_id."""
        raise NotImplementedError

    def folds(self, settings, utterances) -> tuple:
        """The fold scheme for these settings (affect.data.splits Fold or RoleFold objects)."""
        raise NotImplementedError

    def fold_names(self, settings) -> tuple:
        """The names of the scheme's folds, in order, without reading any data (checks sweep files)."""
        raise NotImplementedError
