
yrs = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
yrs = [2019, 2020, 2021, 2022]
yrs = [2018, 2019, 2020]#, 2021, 2022, 2023, 2024]
yrs = [2018, 2020, 2021, 2022, 2023, 2024]
yrs = [2019]
yrs = [2022]
yrs = [2018, 2019, 2021, 2022, 2023, 2024]
#yrs = [2018]
days = []
dn = 0
yrs = [2024]
dn = 1
#while dn < 300:
while dn < 365:
    start = dn
    dn = dn + 60
    if dn > 365:
        dn = 365
    days.append((start,dn))
    dn = dn + 1
print(days)
#print("sbatch --export=START_DN=87,END_DN=87,YEAR=2021 compute.script")
#print("sbatch --export=START_DN=97,END_DN=97,YEAR=2022 head.script")
#print("bash two_nodes.script --start={} --end={} --year={}".format(d[0], d[1], yr))
#print("sbatch --export=START_DN={},END_DN={},YEAR={} head.script".format(d[0], d[1], yr))
for yr in yrs:
    for d in days:
        #print("bash two_nodes.script --start={} --end={} --year={}".format(d[0], d[1], yr))
        #print("sbatch --export=START_DN={},END_DN={},YEAR={} run_Mie.script".format(d[0], d[1], yr))
        #print("sbatch --export=START_DN={},END_DN={},YEAR={} fge_compute.script".format(d[0], d[1], yr))
        #print("sbatch --export=START_DN={},END_DN={},YEAR={} run_pseudo.script".format(d[0], d[1], yr))
        print("sbatch --export=START_DN={},END_DN={},YEAR={} head.script".format(d[0], d[1], yr))

