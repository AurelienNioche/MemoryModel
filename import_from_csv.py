import os
import pandas as pd
from tqdm import tqdm

import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "MemoryModel.settings")
django.setup()

from data_interface.models import Data


class Importer:

    BULK_LIMIT = 10000

    def __init__(self, data_file):

        self.data_file = data_file

        self.entries = []
        self.counter = 0

        self.df = None

    def _run(self):

        print("Deleting previous entries...", end=" ", flush=True)
        Data.objects.all().delete()
        print("Done!")

        self._import_data_file()

    def _import_data_file(self):

        print(f"Reading from '{self.data_file}'...", end=" ", flush=True)
        self.df = pd.read_csv(self.data_file)
        print("Done!")
        n = len(self.df)

        print("Preprocessing and writing the data...", end=" ", flush=True)

        for i in tqdm(range(n)):
            self._import_line(i=i)

        print("Done!")

    def _write_in_db(self):

        Data.objects.bulk_create(self.entries)
        self.entries = []
        self.counter = 0

    def _import_line(self, i):

        e = Data(
            user_id=self.df["user_id"][i],
            item_id=self.df["module_id"][i],
            user_item_pair_id=self.df["student_id"][i],
            outcome=self.df["outcome"][i],
            timestamp=self.df["timestamp"][i],
            t_last=self.df["tlast"][i],
            n_rep=self.df["nreps"][i],
            deck=self.df["deck"][i])

        self._add_to_entries(e)

    def _add_to_entries(self, new_entry):

        self.entries.append(new_entry)
        self.counter += 1
        if self.counter > self.BULK_LIMIT:
            self._write_in_db()

    @classmethod
    def import_from_csv(cls, data_file):

        imp = cls(data_file=data_file)
        imp._run()


def main():

    Importer.import_from_csv(
        data_file=os.path.join("data", "mnemosyne-history.csv"))


if __name__ == "__main__":
    main()