yr = 2024
#while dn < 365:
#    start = dn
#    end = dn + 28
#    print("sbatch --export=START={},END={},YEAR={} run.script;".format(start, end, yr))
#    dn = end

yrs = [2018, 2019, 2020, 2021, 2022, 2023]
yrs = [2024]
months = list(range(1,13))
band = 'C16'
for yr in yrs:
    for month in months:
        print("sbatch --export=MONTH={},YEAR={},BAND={} run.script;".format(month, yr, band))

