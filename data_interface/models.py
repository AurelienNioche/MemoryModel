from django.db import models

# Create your models here.
from django.db import models


class Data(models.Model):

    user_id = models.TextField(db_index=True, blank=True, null=True)
    item_id = models.TextField(db_index=True, blank=True, null=True)
    user_item_pair_id = models.TextField(db_index=True, blank=True, null=True)
    timestamp = models.IntegerField(db_index=True, blank=True, null=True)
    n_rep = models.IntegerField(db_index=True, blank=True, null=True)
    outcome = models.BooleanField(db_index=True, blank=True, null=True)
    t_last = models.IntegerField(db_index=True, blank=True, null=True)
    deck = models.IntegerField(db_index=True, blank=True, null=True)
    delta_last = models.IntegerField(db_index=True, blank=True, null=True)
    n_obs_user = models.IntegerField(db_index=True, blank=True, null=True)
    n_obs_item = models.IntegerField(db_index=True, blank=True, null=True)
    n_obs_user_item_pair = models.IntegerField(db_index=True, blank=True, null=True)

    class Meta:
        db_table = 'data'
