import os

def remove_files_with_exact_filename(root_directory, exact_filename):
    """
    Recursively walk through 'root_directory' and remove files
    whose *file name* matches 'exact_filename' exactly.
    """
    for current_path, dirs, files in os.walk(root_directory):
        for filename in files:
            if filename == exact_filename:
                file_path = os.path.join(current_path, filename)
                try:
                    os.remove(file_path)
                    print(f"Removed file: {file_path}")
                except Exception as e:
                    print(f"Error removing file {file_path}: {e}")

if __name__ == "__main__":
    
    directories = [
        #"../output/main",
        "../output/main/time_attribution/2010",
        "../output/main/time_attribution/2100/SSP1-26",
        "../output/main/time_attribution/2100/SSP5-85",
        "../output/main/time_attribution/past_and_future/SSP1-26",
        "../output/main/time_attribution/past_and_future/SSP5-85",
        "../output/sensitivity/contour_plot/",
        "../output/sensitivity/geogenic_atm_high/",
        "../output/sensitivity/geogenic_atm_low/",
        "../output/sensitivity/geogenic_ocd_high/",
        "../output/sensitivity/geogenic_ocd_low/",
        "../output/sensitivity/high_mobility/",
        "../output/sensitivity/low_mobility/",
        "../output/sensitivity/respiration_high/",
        "../output/sensitivity/respiration_low/",
        "../output/sensitivity/Shah_2021_air_sea/",
        "../output/sensitivity/streets_high/",
        "../output/sensitivity/streets_low/",
        "../output/sensitivity/previous_protected_soil/",
        "../output/sensitivity/previous_ocean/",
        ]
    
    for search_dir in directories:
        for scenario in ["SSP1-26", "SSP5-85"]:
            for year in [-2000, 2010, 2100, 2300]:
                fn = f"rate_matrix_sector_{scenario}_1510_{year}.csv"
                remove_files_with_exact_filename(search_dir, fn)
                fn = f"fragments_sector_{scenario}_1510_{year}.csv"
                remove_files_with_exact_filename(search_dir, fn)
                fn = f"output_sector_{scenario}_1510_{year}.csv"
                remove_files_with_exact_filename(search_dir, fn)
                fn = f"rate_table.csv"
                remove_files_with_exact_filename(search_dir, fn)
                fn = "all_inputs_output_sector_{scenario}_1510_2300.csv"
                remove_files_with_exact_filename(search_dir, fn)

                for agg in [0, 1]:
                    fn = f"budget_table_{year}_agg{agg}_sector_{scenario}_1510_2300.csv"
                    remove_files_with_exact_filename(search_dir, fn)
                    for yr1, yr2 in zip([1509, 1510, 1600, 1700, 1800, 1900, 2000, 2010, 1510], [1510, 1600, 1700, 1800, 1900, 2000, 2010, 2300, 2300]):
                        fn = f"budget_table_{year}_agg{agg}_sector_{scenario}_{yr1}_{yr2}.csv"
                        remove_files_with_exact_filename(search_dir, fn)
                        fn = f"all_inputs_output_sector_{scenario}_{yr1}_{yr2}.csv"
                        remove_files_with_exact_filename(search_dir, fn)
                        fn = f"constraint_table_sector_{scenario}_{yr1}_{yr2}.csv"
                        remove_files_with_exact_filename(search_dir, fn)
                        fn = f"rate_matrix_sector_{scenario}_{yr1}_{yr2}.csv"
                        remove_files_with_exact_filename(search_dir, fn)