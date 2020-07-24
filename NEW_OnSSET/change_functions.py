#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:11:11 2020

@author: balderrama
"""

if end_year_pop == 0:
            onsseter.df[SET_POP + "{}".format(year)] = onsseter.df[SET_POP + "{}".format(year) + 'Low']
else:
            onsseter.df[SET_POP + "{}".format(year)] = onsseter.df[SET_POP + "{}".format(year) + 'High']

if year - time_step == start_year:
            # Assign new connections to those that are already electrified to a certain percent
            onsseter.df.loc[(onsseter.df[SET_ELEC_FUTURE_ACTUAL + "{}".format(
                year - time_step)] == 1), SET_NEW_CONNECTIONS + "{}".format(year)] = \
                (onsseter.df[SET_POP + "{}".format(year)] - onsseter.df[SET_ELEC_POP])
            # Assign new connections to those that are not currently electrified
            onsseter.df.loc[onsseter.df[SET_ELEC_FUTURE_ACTUAL + "{}".format(year - time_step)] == 0,
                        SET_NEW_CONNECTIONS + "{}".format(year)] = onsseter.df[SET_POP + "{}".format(year)]
            # Some conditioning to eliminate negative values if existing by mistake
            onsseter.df.loc[onsseter.df[SET_NEW_CONNECTIONS + "{}".format(year)] < 0,
                        SET_NEW_CONNECTIONS + "{}".format(year)] = 0



onsseter.df['PerCapitaDemand'] = 0

# RUN_PARAM: This shall be changed if different urban/rural categorization is decided
# Create new columns assigning number of people per household as per Urban/Rural type
onsseter.df.loc[onsseter.df[SET_URBAN] == 0, SET_NUM_PEOPLE_PER_HH] = num_people_per_hh_rural
onsseter.df.loc[onsseter.df[SET_URBAN] == 1, SET_NUM_PEOPLE_PER_HH] = num_people_per_hh_rural
onsseter.df.loc[onsseter.df[SET_URBAN] == 2, SET_NUM_PEOPLE_PER_HH] = num_people_per_hh_urban




onsseter.df['PerCapitaDemand'] = 0

            # RUN_PARAM: This shall be changed if different urban/rural categorization is decided
            # Create new columns assigning number of people per household as per Urban/Rural type
onsseter.df.loc[onsseter.df[SET_URBAN] == 0, SET_NUM_PEOPLE_PER_HH] = num_people_per_hh_rural
onsseter.df.loc[onsseter.df[SET_URBAN] == 1, SET_NUM_PEOPLE_PER_HH] = num_people_per_hh_rural
onsseter.df.loc[onsseter.df[SET_URBAN] == 2, SET_NUM_PEOPLE_PER_HH] = num_people_per_hh_urban




            # Define per capita residential demand
            # onsseter.df['PerCapitaDemand'] = onsseter.df['ResidentialDemandTier1.' + str(wb_tier_urban)]
onsseter.df.loc[onsseter.df[SET_URBAN] == 0, 'PerCapitaDemand'] = onsseter.df[
                'ResidentialDemandTier' + str(wb_tier_rural)]
onsseter.df.loc[onsseter.df[SET_URBAN] == 1, 'PerCapitaDemand'] = onsseter.df[
                'ResidentialDemandTier' + str(wb_tier_urban_clusters)]
onsseter.df.loc[onsseter.df[SET_URBAN] == 2, 'PerCapitaDemand'] = onsseter.df[
                'ResidentialDemandTier' + str(wb_tier_urban_centers)]
            # if max(onsseter.df[SET_URBAN]) == 2:
            #     onsseter.df.loc[onsseter.df[SET_URBAN] == 2, 'PerCapitaDemand'] = onsseter.df['ResidentialDemandTier1.' + str(wb_tier_urban_center)]

            # Add commercial demand
            # agri = True if 'y' in input('Include agrcultural demand? <y/n> ') else False
            # if agri:
if int(productive_demand) == 1:
                onsseter.df['PerCapitaDemand'] += onsseter.df['AgriDemand']

            # commercial = True if 'y' in input('Include commercial demand? <y/n> ') else False
            # if commercial:
if int(productive_demand) == 1:
                onsseter.df['PerCapitaDemand'] += onsseter.df['CommercialDemand']

            # health = True if 'y' in input('Include health demand? <y/n> ') else False
            # if health:
if int(productive_demand) == 1:
                onsseter.df['PerCapitaDemand'] += onsseter.df['HealthDemand']

            # edu = True if 'y' in input('Include educational demand? <y/n> ') else False
            # if edu:
if int(productive_demand) == 1:
                onsseter.df['PerCapitaDemand'] += onsseter.df['EducationDemand']

onsseter.df.loc[onsseter.df[SET_URBAN] == 0, SET_ENERGY_PER_CELL + "{}".format(year)] = \
            onsseter.df['PerCapitaDemand'] * onsseter.df[SET_NEW_CONNECTIONS + "{}".format(year)]

onsseter.df.loc[onsseter.df[SET_URBAN] == 1, SET_ENERGY_PER_CELL + "{}".format(year)] = \
            onsseter.df['PerCapitaDemand'] * onsseter.df[SET_NEW_CONNECTIONS + "{}".format(year)]

onsseter.df.loc[onsseter.df[SET_URBAN] == 2, SET_ENERGY_PER_CELL + "{}".format(year)] = \
            onsseter.df['PerCapitaDemand'] * onsseter.df[SET_NEW_CONNECTIONS + "{}".format(year)]

        # if year - time_step == start_year:
onsseter.df.loc[onsseter.df[SET_URBAN] == 0, SET_TOTAL_ENERGY_PER_CELL] = \
            onsseter.df['PerCapitaDemand'] * onsseter.df[SET_POP + "{}".format(year)]

onsseter.df.loc[onsseter.df[SET_URBAN] == 1, SET_TOTAL_ENERGY_PER_CELL] = \
            onsseter.df['PerCapitaDemand'] * onsseter.df[SET_POP + "{}".format(year)]

onsseter.df.loc[onsseter.df[SET_URBAN] == 2, SET_TOTAL_ENERGY_PER_CELL] = \
            onsseter.df['PerCapitaDemand'] * onsseter.df[SET_POP + "{}".format(year)]



























