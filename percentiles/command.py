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
sat = 'G18'
bands = ['C14', 'C15', 'C16']
for yr in yrs:
    for band in bands:
        for month in months:
            print("sbatch --export=SAT={},BAND={},YEAR={},MONTH={} run_download.script;".format(sat, band, yr, month))
        #print("sbatch --export=SAT={},BAND={},YEAR={},MONTH={} run_download.script;".format(sat, band, yr, 'all'))
        #print("sbatch --export=SAT={},BAND={},YEAR={},MONTH={} run_stats.script;".format(sat, band, yr, month))

