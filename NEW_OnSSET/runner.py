# Pulls all the other functions together to make magic!
#
# Author: KTH dESA Last modified by Alexandros Korkovelos
# Date: 07 February 2019
# Python version: 3.5

import os
from onsset import *
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from openpyxl import load_workbook

root = tk.Tk()
root.withdraw()
root.attributes("-topmost", True)

messagebox.showinfo('OnSSET', 'Open the specs file')
specs_path = filedialog.askopenfilename()

# specs = pd.read_excel(specs_path, index_col=0)

# RUN_PARAM: Insert the name of the country you are working on. More countries should be separated using comma e.g. ["Malawi", "Ghana"]
countries = ['Bolivia']
#countries = str(input('countries: ')).split()
# countries = specs.index.tolist() if 'all' in countries else countries

choice = int(input('Enter 1 to split (if multiple country run is needed), 2 to prepare/calibrate the GIS input file, 3 to run scenario(s): '))

# TODO Do we actually need option 1 anymore? I suggest removing it and readjust the options
if choice == 1:
    messagebox.showinfo('OnSSET', 'Open the csv file with GIS data')
    settlements_csv = filedialog.askopenfilename()
    messagebox.showinfo('OnSSET', 'Select the folder to save split countries')
    base_dir = filedialog.asksaveasfilename()

    print('\n --- Splitting --- \n')

    df = pd.read_csv(settlements_csv)

    for country in countries:
        print(country)
        df.loc[df[SET_COUNTRY] == country].to_csv(base_dir + '.csv', index=False)

elif choice == 2:
    SpecsData = pd.read_excel(specs_path, sheet_name='SpecsData') ######################Check all the sheet_name, for some reason some times uses sheetname and not sheet_name!!!!!!!!!
    messagebox.showinfo('OnSSET', 'Open the file containing separated countries')
    base_dir = filedialog.askopenfilename()
    messagebox.showinfo('OnSSET', 'Browse to result folder and name the calibrated file')
    output_dir = filedialog.asksaveasfilename()

    print('\n --- Prepping --- \n')

    for country in countries:
        print(country)
        settlements_in_csv = base_dir # os.path.join(base_dir, '{}.csv'.format(country))
        settlements_out_csv = output_dir + '.csv' # os.path.join(output_dir, '{}.csv'.format(country))

        onsseter = SettlementProcessor(settlements_in_csv)

        num_people_per_hh_rural = float(SpecsData.iloc[0][SPE_NUM_PEOPLE_PER_HH_RURAL])
        num_people_per_hh_urban = float(SpecsData.iloc[0][SPE_NUM_PEOPLE_PER_HH_URBAN])

        onsseter.prepare_wtf_tier_columns(num_people_per_hh_rural, num_people_per_hh_urban)
        onsseter.condition_df(country)
        onsseter.grid_penalties()
        onsseter.calc_wind_cfs()

        pop_actual = SpecsData.loc[0, SPE_POP]
        pop_future_high = SpecsData.loc[0, SPE_POP_FUTURE + 'High']
        pop_future_low = SpecsData.loc[0, SPE_POP_FUTURE + 'Low']
        urban_current = SpecsData.loc[0, SPE_URBAN]
        urban_future = SpecsData.loc[0, SPE_URBAN_FUTURE]
        urban_cutoff = SpecsData.loc[0, SPE_URBAN_CUTOFF]
        start_year = int(SpecsData.loc[0, SPE_START_YEAR])
        end_year = int(SpecsData.loc[0, SPE_END_YEAR])

        elec_actual = SpecsData.loc[0, SPE_ELEC]
        elec_actual_urban = SpecsData.loc[0, SPE_ELEC_URBAN]
        elec_actual_rural = SpecsData.loc[0, SPE_ELEC_RURAL]
        pop_cutoff = SpecsData.loc[0, SPE_POP_CUTOFF1]
        min_night_lights = SpecsData.loc[0, SPE_MIN_NIGHT_LIGHTS]
        max_grid_dist = SpecsData.loc[0, SPE_MAX_GRID_DIST]
        max_road_dist = SpecsData.loc[0, SPE_MAX_ROAD_DIST]
        pop_tot = SpecsData.loc[0, SPE_POP]
        pop_cutoff2 = SpecsData.loc[0, SPE_POP_CUTOFF2]
        dist_to_trans = SpecsData.loc[0, SPE_DIST_TO_TRANS]

        urban_cutoff, urban_modelled = onsseter.calibrate_pop_and_urban(pop_actual, pop_future_high, pop_future_low, urban_current,
                                                                        urban_future, urban_cutoff, start_year, end_year)

        min_night_lights, dist_to_trans, max_grid_dist, max_road_dist, elec_modelled, pop_cutoff, pop_cutoff2, rural_elec_ratio, urban_elec_ratio = \
            onsseter.elec_current_and_future(elec_actual, elec_actual_urban, elec_actual_rural, pop_cutoff, dist_to_trans, min_night_lights, max_grid_dist,
                                             max_road_dist, pop_tot, pop_cutoff2, start_year)

        # In case there are limitations in the way grid expansion is moving in a country, this can be reflected through gridspeed.
        # In this case the parameter is set to a very high value therefore is not taken into account.
        onsseter.grid_reach_estimate(start_year, gridspeed=9999)

        SpecsData.loc[0, SPE_URBAN_MODELLED] = urban_modelled
        SpecsData.loc[0, SPE_URBAN_CUTOFF] = urban_cutoff
        SpecsData.loc[0, SPE_MIN_NIGHT_LIGHTS] = min_night_lights
        SpecsData.loc[0, SPE_MAX_GRID_DIST] = max_grid_dist
        SpecsData.loc[0, SPE_MAX_ROAD_DIST] = max_road_dist
        SpecsData.loc[0, SPE_ELEC_MODELLED] = elec_modelled
        SpecsData.loc[0, SPE_POP_CUTOFF1] = pop_cutoff
        SpecsData.loc[0, SPE_POP_CUTOFF2] = pop_cutoff2
        SpecsData.loc[0, 'rural_elec_ratio_modelled'] = rural_elec_ratio
        SpecsData.loc[0, 'urban_elec_ratio_modelled'] = urban_elec_ratio

        book = load_workbook(specs_path)
        writer = pd.ExcelWriter(specs_path, engine='openpyxl')
        writer.book = book
        # RUN_PARAM: Here the calibrated "specs" data are copied to a new tab called "SpecsDataCalib". This is what will later on be used to feed the model
        SpecsData.to_excel(writer, sheet_name='SpecsDataCalib', index=False)
        writer.save()
        writer.close()

        logging.info('Calibration finished. Results are transferred to the csv file')
        onsseter.df.to_csv(settlements_out_csv, index=False)
# 11124324.369249057
elif choice == 3:

    diesel_high = True
    diesel_tag = 'high' if diesel_high else 'low'

    messagebox.showinfo('OnSSET', 'Open the csv file with calibrated GIS data')
    base_dir = filedialog.askopenfilename()
    messagebox.showinfo('OnSSET', 'Browse to result folder and name the scenario to save outputs')
    # output_dir = filedialog.asksaveasfilename()
    output_dir = filedialog.askdirectory()

    print('\n --- Running scenario --- \n')

    ScenarioInfo = pd.read_excel(specs_path, sheet_name='ScenarioInfo')
    Scenarios = ScenarioInfo['Scenario']
    ScenarioParameters = pd.read_excel(specs_path, sheet_name='ScenarioParameters')
    SpecsData = pd.read_excel(specs_path, sheet_name='SpecsDataCalib')

    for scenario in Scenarios:
        print('Scenario: ' + str(scenario+1))
        countryID = SpecsData.iloc[0]['CountryCode']

        popIndex = ScenarioInfo.iloc[scenario]['Population_Growth']
        tierIndex = ScenarioInfo.iloc[scenario]['Target_electricity_consumption_level']
        fiveyearIndex = ScenarioInfo.iloc[scenario]['Electrification_target_5_years']
        gridIndex = ScenarioInfo.iloc[scenario]['Grid_electricity_generation_cost']
        pvIndex = ScenarioInfo.iloc[scenario]['PV_cost_adjust']
        dieselIndex = ScenarioInfo.iloc[scenario]['Diesel_price']
        productiveIndex = ScenarioInfo.iloc[scenario]['Productive_uses_demand']
        prioIndex = ScenarioInfo.iloc[scenario]['Prioritization_algorithm']

        end_year_pop = ScenarioParameters.iloc[popIndex]['PopEndYear']
        rural_tier = ScenarioParameters.iloc[tierIndex]['RuralTargetTier']
        urban_tier = ScenarioParameters.iloc[tierIndex]['UrbanTargetTier']
        five_year_target = ScenarioParameters.iloc[fiveyearIndex]['5YearTarget']
        grid_price = ScenarioParameters.iloc[gridIndex]['GridGenerationCost']
        pv_capital_cost_adjust = ScenarioParameters.iloc[pvIndex]['PV_Cost_adjust']
        diesel_price = ScenarioParameters.iloc[dieselIndex]['DieselPrice']
        productive_demand = ScenarioParameters.iloc[productiveIndex]['ProductiveDemand']
        prioritization = ScenarioParameters.iloc[prioIndex]['PrioritizationAlgorithm']

        settlements_in_csv = base_dir # os.path.join(base_dir, '{}.csv'.format(country))
        # settlements_out_csv = output_dir + '.csv' # os.path.join(output_dir, '{}_{}_{}.csv'.format(country, wb_tier_urban, diesel_tag))
        settlements_out_csv = os.path.join(output_dir, '{}-1_{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(countryID, popIndex, tierIndex, fiveyearIndex, gridIndex, pvIndex, dieselIndex, productiveIndex, prioIndex))
        summary_csv = os.path.join(output_dir, '{}-1_{}_{}_{}_{}_{}_{}_{}_{}_summary.csv'.format(countryID, popIndex, tierIndex, fiveyearIndex, gridIndex, pvIndex, dieselIndex, productiveIndex, prioIndex))
        # summary_csv = output_dir + 'summary.csv'

        onsseter = SettlementProcessor(settlements_in_csv)

        start_year = SpecsData.iloc[0][SPE_START_YEAR]
        end_year = SpecsData.iloc[0][SPE_END_YEAR]

        existing_grid_cost_ratio = SpecsData.iloc[0][SPE_EXISTING_GRID_COST_RATIO]
        num_people_per_hh_rural = float(SpecsData.iloc[0][SPE_NUM_PEOPLE_PER_HH_RURAL])
        num_people_per_hh_urban = float(SpecsData.iloc[0][SPE_NUM_PEOPLE_PER_HH_URBAN])
        max_grid_extension_dist = float(SpecsData.iloc[0][SPE_MAX_GRID_EXTENSION_DIST])
        urban_elec_ratio = float(SpecsData.iloc[0]['rural_elec_ratio_modelled'])
        rural_elec_ratio = float(SpecsData.iloc[0]['urban_elec_ratio_modelled'])
        grid_cap_gen_limit = SpecsData.loc[0, 'NewGridGenerationCapacityTimestepLimit']*1000

        # RUN_PARAM: Fill in general and technology specific parameters (e.g. discount rate, losses etc.)
        Technology.set_default_values(base_year=start_year,
                                      start_year=start_year,
                                      end_year=end_year,
                                      discount_rate=0.12)

        grid_calc = Technology(om_of_td_lines=0.02,
                               distribution_losses=float(SpecsData.iloc[0][SPE_GRID_LOSSES]),
                               connection_cost_per_hh=125,
                               base_to_peak_load_ratio=float(SpecsData.iloc[0][SPE_BASE_TO_PEAK]),
                               capacity_factor=1,
                               tech_life=30,
                               grid_capacity_investment=float(SpecsData.iloc[0][SPE_GRID_CAPACITY_INVESTMENT]),
                               grid_penalty_ratio=1,
                               grid_price=grid_price)

        mg_hydro_calc = Technology(om_of_td_lines=0.02,
                                   distribution_losses=0.05,
                                   connection_cost_per_hh=125,
                                   base_to_peak_load_ratio=1,
                                   capacity_factor=0.5,
                                   tech_life=30,
                                   capital_cost=5000,
                                   om_costs=0.02)

        mg_wind_calc = Technology(om_of_td_lines=0.02,
                                  distribution_losses=0.05,
                                  connection_cost_per_hh=125,
                                  base_to_peak_load_ratio=0.75,
                                  capital_cost=2500,
                                  om_costs=0.02,
                                  tech_life=20)

        mg_pv_calc = Technology(om_of_td_lines=0.03,
                                distribution_losses=0.05,
                                connection_cost_per_hh=125,
                                base_to_peak_load_ratio=0.9,
                                tech_life=20,
                                om_costs=0.02,
                                capital_cost=3500 * pv_capital_cost_adjust)

        sa_pv_calc = Technology(base_to_peak_load_ratio=0.9,
                                tech_life=15,
                                om_costs=0.02,
                                capital_cost={0.020: 20000, 0.050: 11050, 0.100: 
                                    7660, 0.200: 5780, 0.300: 5070},
                                #capital_cost=5500 * pv_capital_cost_adjust,
                                standalone=True)

        mg_diesel_calc = Technology(om_of_td_lines=0.02,
                                    distribution_losses=0.05,
                                    connection_cost_per_hh=125,
                                    base_to_peak_load_ratio=0.5,
                                    capacity_factor=0.7,
                                    tech_life=15,
                                    om_costs=0.1,
                                    efficiency=0.33,
                                    capital_cost=1000,
                                    diesel_price=diesel_price,
                                    diesel_truck_consumption=33.7,
                                    diesel_truck_volume=15000)

        sa_diesel_calc = Technology(base_to_peak_load_ratio=0.5,
                                    capacity_factor=0.5,
                                    tech_life=10,
                                    om_costs=0.1,
                                    capital_cost=938,
                                    diesel_price=diesel_price,
                                    standalone=True,
                                    efficiency=0.28,
                                    diesel_truck_consumption=14,
                                    diesel_truck_volume=300)
        
        
        HighLands = location(name='HighLands', altitude = [799.999999,7000])
        Amazonia = location(name='Amazonia', altitude = [-0.1,799.999999], 
                            latitude=[-17.999999,0] )
        Chaco = location(name='Chaco', altitude = [-0.1,799.9999999], latitude=[-90,-18])
        
        
        # RUN_PARAM: Activating (un-commenting) lines 254-294 will run the analysis without time step and help identify differences in the two modelling approaches
        ### RUN - NO TIMESTEP

        # # RUN_PARAM: Fill in the next 3 parameters accordingly. Remember this specifies a run with no intermediate step
        # time_step = 13            # Years between final and start year
        # year = 2025               # Final year
        # eleclimits = {2025: 1}    # Access goal in the final year
        #
        # eleclimit = eleclimits[year]
        #
        # onsseter.set_scenario_variables(year, num_people_per_hh_rural, num_people_per_hh_urban, time_step, start_year,
        #                                 urban_elec_ratio, rural_elec_ratio, urban_tier, rural_tier, end_year_pop, productive_demand)
        #
        #
        # onsseter.calculate_off_grid_lcoes(mg_hydro_calc, mg_wind_calc, mg_pv_calc, sa_pv_calc, mg_diesel_calc,
        #                                   sa_diesel_calc, year, start_year, end_year, time_step)
        #
        # if year - time_step == start_year:
        #     onsseter.current_mv_line_dist()
        #
        # onsseter.pre_electrification(grid_calc, grid_price, year, time_step, start_year)
        #
        #
        #
        # onsseter.run_elec(grid_calc, max_grid_extension_dist, year, start_year, end_year, time_step, grid_cap_gen_limit)
        #
        # onsseter.results_columns(mg_hydro_calc, mg_wind_calc, mg_pv_calc, sa_pv_calc, mg_diesel_calc, sa_diesel_calc,
        #                          grid_calc, year)
        #
        # onsseter.calculate_investments(mg_hydro_calc, mg_wind_calc, mg_pv_calc, sa_pv_calc, mg_diesel_calc,
        #                sa_diesel_calc, grid_calc, year, end_year, time_step)
        #
        # onsseter.apply_limitations(eleclimit, year, time_step, prioritization)
        #
        # onsseter.final_decision(mg_hydro_calc, mg_wind_calc, mg_pv_calc, sa_pv_calc, mg_diesel_calc, sa_diesel_calc,
        #                         grid_calc, year, end_year, time_step)
        #
        # onsseter.delete_redundant_columns(year)

        ### END OF FIRST RUN

        ### HERE STARTS THE ACTUAL ANALYSIS WITH THE INCLUSION OF TIME STEPS


        # RUN_PARAM: One shall define here the years of analysis (excluding start year) together with access targets per interval and timestep duration
        yearsofanalysis = [2025]
        eleclimits = {2025: 0.97}
        time_steps = {2025: 13}

        elements = ["1.Population", "2.New_Connections", "3.Capacity", "4.Investment"]
        techs = ["Grid", "SA_Diesel", "SA_PV", "MG_Diesel", "MG_PV", "MG_Wind", "MG_Hydro"]
        
        Places = []
        
        sumtechs = []

        for element in elements:
            for tech in techs:
                sumtechs.append(element + "_" + tech)

        sumtechs.append('Min_cluster_pop_2030')
        sumtechs.append('Max_cluster_pop_2030')
        sumtechs.append('Min_cluster_area')
        sumtechs.append('Max_cluster_area')
        sumtechs.append('Min_existing_grid_dist')
        sumtechs.append('Max_existing_grid_dist')
        sumtechs.append('Min_road_dist')
        sumtechs.append('Max_road_dist')
        sumtechs.append('Min_investment_capita_cost')
        sumtechs.append('Max_investment_capita_cost')

        total_rows = len(sumtechs)

        df_summary = pd.DataFrame(columns=yearsofanalysis)

        for row in range(0, total_rows):
            df_summary.loc[sumtechs[row]] = "Nan"

        onsseter.current_mv_line_dist()

        for year in yearsofanalysis:

            #eleclimit = float(input('Provide the targeted electrification rate in {}:'.format(year)))
            eleclimit = eleclimits[year]
            time_step = time_steps[year]
            #investlimit = int(input('Provide the targeted investment limit (in USD) for the year {}:'.format(year)))
            #Sets
            #MVConnectDist Pop2025	NewConnections2025	
            #NumPeoplePerHH	EnergyPerSettlement2025	TotalEnergyPerCell

            onsseter.set_scenario_variables(year, num_people_per_hh_rural, num_people_per_hh_urban, time_step,
                                            start_year, urban_elec_ratio, rural_elec_ratio, urban_tier,
                                            rural_tier, end_year_pop, productive_demand)

            #Sets
            #MG_PV2025	MG_Wind2025	MG_Diesel2025	SA_Diesel2025	
            #SA_PV2025	Minimum_Tech_Off_grid2025	Minimum_LCOE_Off_grid2025	
            #Off_Grid_Code2025

            onsseter.calculate_off_grid_lcoes(mg_hydro_calc, mg_wind_calc, mg_pv_calc, sa_pv_calc, 
                                              mg_diesel_calc, sa_diesel_calc, year, start_year, end_year, time_step)

            #Sets
            # Elec_Initial_Status_Grid2025	Elec_Init_Status_Offgrid2025	
            #Actual_Elec_Status_2025	Grid_extension2025

            
            onsseter.pre_electrification(grid_calc, grid_price, year, time_step, start_year)
                        
            #Sets
            # MinGridDist2025	ElectrificationOrder2025
            # it changes the Grid_extension2025

            onsseter.run_elec(grid_calc, max_grid_extension_dist, year, start_year,
                              end_year, time_step, grid_cap_gen_limit)
            
            #Sets
            # MinimumOverallLCOE2025	MinimumOverallCode2025


            onsseter.results_columns(mg_hydro_calc, mg_wind_calc, mg_pv_calc, sa_pv_calc, mg_diesel_calc,
                                     sa_diesel_calc, grid_calc, year)
            #SETS
            # ElecStatusIn2025	GridElecIn2025	OffGridElecIn2025	
            #FinalElecCode2025	

            
            onsseter.calculate_investments(mg_hydro_calc, mg_wind_calc, mg_pv_calc, sa_pv_calc, mg_diesel_calc,
                                           sa_diesel_calc, grid_calc, year, end_year, time_step)
            
            # Sets
            
            onsseter.apply_limitations(eleclimit, year, time_step, prioritization)
            
            # Sets
            # NewCapacity2025	InvestmentCostLV2025	InvestmentCostMV2025	
            # InvestmentCostHV2025	InvestmentCostTransformer2025	
            # InvestmentCostConnection2025

            
            onsseter.final_decision(mg_hydro_calc, mg_wind_calc, mg_pv_calc, sa_pv_calc,
                                    mg_diesel_calc, sa_diesel_calc, grid_calc, year, end_year, time_step)

            onsseter.calc_summaries(df_summary, sumtechs, year)
            
            onsseter.Demand_Regressions(HighLands, Amazonia, Chaco)
#            
#            
            onsseter.Surrogate_Hybrid(0.12, 20, HighLands, Amazonia, Chaco)
            onsseter.Surrugate_HouseHold(HighLands, Amazonia, Chaco)
            
            
        df_summary.to_csv(summary_csv, index=sumtechs)
        onsseter.df.to_csv(settlements_out_csv, index=False)

df = onsseter.df
Grid_elec = onsseter.df.loc[onsseter.df['ElecStatusIn2025']==1]
Isolated_elec = onsseter.df.loc[onsseter.df['ElecStatusIn2025']!=1]

Micro_Grids = Isolated_elec.loc[Isolated_elec['NPC_Hybrid']!=99]
Home_System = Isolated_elec.loc[Isolated_elec['NPC_HomeSystem']!=99]



print('Grid total cost is ' + str(round(df_summary[2025]['4.Investment_Grid'],0)) + ' USD')
print("Microgrid total cost is " + str(round(Micro_Grids['NPC_Hybrid'].sum(),0)) + ' USD')
print("PV HomeSystems total cost is "
      + str(round(Home_System['HouseHolds'].sum()*1000,0)) + ' USD')


print('Grid cost per household ' + str(round(df_summary[2025]['4.Investment_Grid']/
                                             (df_summary[2025]['2.New_Connections_Grid']/3.6),0)) + ' USD')
print("Cost per Household total cost is " 
      + str(round(Micro_Grids['NPC_Hybrid'].sum() /Micro_Grids['HouseHolds'].sum(),0)) + ' USD')
print("Cost per HomeSystems total cost is 1000 USD")

print('Electrified population with the grid ' + str(round((df_summary[2025]['2.New_Connections_Grid']),0)) + ' inhabitants')
print("Electrified population with hybrid microgrids "  + str(round(Micro_Grids['Pop2025'].sum(),0)) + ' inhabitants')
print("Electrified population with PV homesystems " + str(round(Home_System['Pop2025'].sum(),0)) + ' inhabitants')

print('New Capacity with the grid ' + str(round((df_summary[2025]['3.Capacity_Grid']/1000),0)) + ' MW')
print("New Capacity with hybrid microgrids "  + str(round(Micro_Grids['PV_Hybrid'].sum()/1000,0)) + ' MW')
print("New Capacity with PV homesystems " + str(round((Home_System['HouseHolds'].sum()*0.1)/1000,0)) + ' MW')


