# Choix des variables

## Categorical variables :

- `contextid`
- `display_env`
- `target_env`
- `user_country`
- `campaignscenario`
- `campaignvertical`
- `is_interstitial`
- `device_type`

Liste finale Python : 
```
['contextid',
 'display_env',
 'target_env',
 'user_country', 
 'campaignscenario',
 'campaignvertical',
 'is_interstitial',
 'device_type']
```

## Quantitative variables

- `contextid`

- ~~`googleviewability`~~ et ~~`googlepagevertical`~~ : pas assez de données

- `campaignctrlast24h`

- `dayssincelastvisitdouble`

- ~~`ltf_lastpartnerclicktimestamp`~~ : très fortement correlé à `nbdaysincelastclick` (corr > 0.8)

- ~~`ltf_nbglobalclick_4w`~~ : corrélé à `ltf_nbglobaldisplay_4w` est valeurs plus faibles.

- `ltf_nbglobaldisplay_4w`

- ~~`ltf_nbglobaldisplaysincelastpartnerproductview`~~  : corrélé à `dayssincelastvisitdouble` et `ltf_nbglobaldisplay_4w`

- `ltf_nbpartnerdisplayssincelastclick`

- ~~`ltf_nbpartnerclick_4w`~~ : fortement corrélé à `ltf_nbpartnerclick_90d` (corr > 0.8) et valeurs plus faibles (4w < 90d)

- ~~`ltf_nbpartnerdisplay_4w`~~ : fortement corrélé à `ltf_nbpartnerdisplay_90d` (corr > 0.8) et valeurs plus faibles (4w < 90d)

- ~~`ltf_nbpartnersales_4w`~~ : fortement corrélé à `ltf_nbpartnersales_90d` (corr > 0.8) et valeurs plus faibles (4w < 90d)

- `ltf_nbpartnerdisplay_90d`

- `ltf_nbpartnerclick_90d`

- `ltf_nbpartnersales_90d`

- `nbdayssincelastclick`

- `nbdisplay_1hour`

- ~~`nbdisplaypartnerapprox_1d_sum_xdevice`~~ : corrélé à `nbdisplay_1hour` et à `nbdisplayglobalapprox_1d_sum_xdevice`

- ~~`nbdisplayaffiliateapprox_1d_sum_xdevice`~~ : corrélé à `ltf_nbglobaldisplay_4w` et fortemet corrélé à `nbdisplayglobalapprox_1d_sum_xdevice` (corr > 0.8)

- `nbdisplayglobalapprox_1d_sum_xdevice`

- `valueperclick`,

- ~~`display_width`~~ et ~~`display_height`~~ : remplacés par `display_size`

- `display_size`

- ~~`display_timestamp`~~

- ~~`is_display_clicked`~~ : variable à prédire

- `zonecostineuro`

Liste finale Python : 
```
['contextid',
 'campaignctrlast24h',
 'dayssincelastvisitdouble',
 'ltf_nbglobaldisplay_4w',
 'ltf_nbpartnerdisplayssincelastclick',
 'ltf_nbpartnerdisplay_90d',
 'ltf_nbpartnerclick_90d',
 'ltf_nbpartnersales_90d',
 'nbdayssincelastclick',
 'nbdisplay_1hour',
 'nbdisplayglobalapprox_1d_sum_xdevice',
 'valueperclick',
 'display_size',
 'zonecostineuro']
```



]
