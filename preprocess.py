import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "MemoryModel.settings")
django.setup()

import numpy as np
from tqdm import tqdm
from django.utils import timezone

from data_interface.models import Data


def main():

    entries = Data.objects.all()

    keys = ('user_id', 'item_id', 'user_item_pair_id')
    counts = {k: {} for k in keys}
    for k in tqdm(keys):
        a, c = np.unique(entries.values_list(k, flat=True), return_counts=True)
        for i in range(len(a)):
            counts[k][a[i]] = c[i]

    to_insert = []
    fields = ['n_obs_user', 'n_obs_item', 'n_obs_user_item_pair', 'delta_last']

    # n = entries.count()
    # i_insert = 0
    print("modifying")
    for e in entries: #, total=n):
        e.delta_last = e.timestamp - e.t_last
        e.n_obs_user = counts["user_id"][e.user_id]
        e.n_obs_item = counts["item_id"][e.item_id]
        e.n_obs_user_item_pair = counts["user_item_pair_id"][e.user_item_pair_id]
        # to_insert.append(e)
        # i_insert += 1
        # if i_insert == 10000:
        #     Data.objects.bulk_update(to_insert, fields)
        #     to_insert = []
        #     i_insert = 0
    print("saving")
    a = timezone.now()
    Data.objects.bulk_update(entries, fields, batch_size=1000)
    b = timezone.now()
    print(b-a)


if __name__ == "__main__":
    main()
