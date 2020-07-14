# data/slurm_runs/92931 9.5G DELETED
This is a run with N = 64, U = 5, J = 5 and 500 temp steps between T = 0.5 and T = 1.5
It has 100,000 steps with thinning = 1 which turns out to be a bit too much data
The point is to provide a test for the reweighting scheme by comparing the reweighting to real data

## data/slurm_runs/93435 407M
This one is exactly the same as the above except Now thining = 100 so there will only be 1000 data points saved. 

## data/slurm_runs/94443
Same as the above but now uses 1000,000 steps and thinning = 1000
NB: accidentally ran this with 500 jobs instead of 100, so ignore the logs for jobs 100-500

## data/slurm_runs/94948
Uses 1000,000 steps and thinning = 1000 but a more interesting temp range.

## data/slurm_runs/95035 1.7G
This is intended to allow the reweighting to be used to estimate the binder crossing. 
Ts = np.linspace(1.0, 2.0, 300) which is slightly denser than the 0.005 T spacing used in 94948
Uses 1000,000 steps and thinning = 1000 as above
Uses Ns = [4,8,16,32,64,128]

Most of these took 6 hours but some upto 16 hours.

## data/slurm_runs/95545_[0-24] 118M DELETED

This is a test of the T-J phase diagram, it should run 25 jobs, covering 5 values of T and 5 of J and each time time do Ns = [4,8,16,32,64,128] and 1000 MCMC steps with no burn in. Should take about 19s per job.

DONE: Took between 0-4 mins because of my random delay settings
      
## data/slurm_runs/95570_[0-399] 2.3G

This is the prelim T-J phase diagram, 400 jobs, covering 20 values of T and 5 of J and each time time do Ns = [4,8,16,32,64,128] and 100,000 MCMC steps with thinning = 100. Should take about 0.6 hours per job.
In reality it took upto 6 or 8 hours per job.
This ran to fruition and looks good. See T-J_plot

The ones with beta=1000 were too cold to run.

## data/slurm_runs/96040 580M
This is intended to produce a plot for a fixed J=5 line.
Ts = np.linspace(0, 10, 100) which is a 0.1 T, coarser than the 0.005 spacing used in 94948
Uses 100,000 steps and thinning = 1000 so should take about 0.6 - 1.6 hours per job.
Uses Ns = [4,8,16,32,64,128]

This one also died at T=0 beta=1/0 for obvious reasons.

## data/slurm_runs/96792 3.2G
This is to add to the T-J plot from 95570 with some finer data around the phase diagram.
It's defined on a region plotted in T-J_plot.ipynb, that has width 7 in T space and uses a spacing of 0.25 for 28 points. J remains 0-20 with 20 points.
Uses 100,000 steps and thinning = 100 so should take about 0.6 - 1.6 hours per job.
Uses Ns = [4,8,16,32,64,128]

## data/slurm_runs/96843 627M #Analysed
This is intended to produce an even more zoomed in plot for a fixed J=5 line.
Ts = np.linspace(1.5, 2.5, 100) which is a 0.01 T, slightly coarser than the 0.005 spacing used in 94948
Uses 1000,000 steps and thinning = 1000 so should take about 6 - 16 hours per job.
Uses Ns = [4,8,16,32,64,128]


## Everyhting above here has been moved to ~/workspace/less_important_slurm_runs

## data/slurm_runs/97469

This is a single run at fixed (J = 5, U = 5, T = 1) replicated 100 times
Ns = [4,8,16,32,64,128]
N_steps = int(100 * thousand),
N_burn_in = int(10 * thousand),
thin = 1 * thousand,

## data/slurm_runs/97801


## data/slurm_runs/98694 5.8G
The above but with T from 0.1 to 10 with 100 steps
I'm repeating each jobs 10 times and reducing the number of steps
The same as 96040 but with U = 1
This is intended to produce a plot for a fixed J=5 line.
Uses 1000,000 steps and thinning = 1000 so should take about 0.6 - 1.6 hours per job.
Uses Ns = [4,8,16,32,64,128]

## data/slurm_runs/99788 data files deleted, data pickled in data.pickle and backed up in data.pickle.backup
This is to make a T-U phase diagram, it's 1000 jobs, covers U and T from 0 to 10 in 10 steps and repeats each jobs 10 times. uses 10,000 steps with thin = 100 giving 1000 samples per point after aggregating the 10 repeats. 
Hopefully the jobs won't take much more than an hour to run. (they didn't, took about 1 hour)
EDIT: I stopped this early to do the next one with a tighter spacing
repeats = 10
Us = np.linspace(0.001, 10.0, 10)
Ts = np.linspace(0.001, 10.0, 10)
J = 5
Ns = [4,8,16,32,64,128]

## data/slurm_runs/100942 data files deleted, data pickled in data.pickle and backed up in data.pickle.backup
This is a zoomed in TU phase diagram
repeats = 2
Us = np.linspace(0.001, 10.0, 20)
Ts = np.linspace(1.0, 3.0, 20)
J = 5
Ns = [4,8,16,32,64,128]

## data/slurm_runs/101765 data files deleted, data pickled in data.pickle and backed up in data.pickle.backup
This is a T-J phase diagram with a different alpha.
repeats = 2
Js = np.linspace(0.001, 10.0, 20)
Ts = np.linspace(0.001, 10.0, 20)
U = 5
alpha = 1.25
Ns = [4,8,16,32,64,128]

## data/slurm_runs/102989
This is just three temperatures, low, critical and high, designed to take a closer look at the correlation functions.
It takes about 38 seconds to run 1000 steps, so 10,000 should take about an hour per job

repeats = 100
J = 5
Ts = np.array([8, 1.8, 1])
U = 5
alpha = 1.25
Ns = [4,8,16,32,64,128,256]
N_steps = int(5*1000),
N_burn_in = int(1*1000), 
thin = 100,

## data/slurm_runs/103867
This is for the band opening plot Figure 2. It's a fixed J=5, U=1, alpha = 1.25, with T from 0.1 to 4 with 200 steps
very fine energy spacing: logger = Eigenspectrum_IPR_all(bins = 2000, limit = 5),
Ns = [4,8,16,32,64,128]

N_steps = int(100*1000),
N_burn_in = int(1*1000), 
thin = 1000

Repeats = 4

## data/slurm_runs/101765 data files deleted, data pickled in data.pickle and backed up in data.pickle.backup
This is a T-J phase diagram with a different alpha.
repeats = 2
Js = np.linspace(0.001, 10.0, 20)
Ts = np.linspace(0.001, 10.0, 20)
U = 5
alpha = 1.25
Ns = [4,8,16,32,64,128]

Plan for figures:
    1) TJ phase diagram, T = linspace(0.1, 5.0, 30) J = linspace(0, 10.0, 30) U = 1 alpha = 1.25 steps = 1e5
        Consider saving a separate file with just the means to keep data size and analusis time down
    
    2) TU phase diagram, T = linspace(0.1, 5.0, 30) U = linspace(0, 10.0, 30) J = 5 alpha = 1.25 steps = 1e5
    
    3) T-alpha phase diagram T = linspace(0.1, 5.0, 30) alpha = linspace(0.5, 4, 30) U = 1, J = 5 steps = 1e5
    
    4) T sweep: T = linspace(0.1, 4.0, 20) J=5, U=1, alpha = 1.25, steps = 1e6
    
    4.5) T sweep for binder crossing: T = linspace(1.8, 2.5, 20) J=5, U=1, alpha = 1.25, steps = 1e6
        Number of steps and grid spacing should be determined based on a measurement of the gradients  and errors from from 4)
        Using sqrt(N) scaling can decide how to scale the spacing and number of samples. 
    
    5) High, Critical, CDW phase and Low temps (T>Tc, T=Tc, Tc>T, T ~ 0)for IPR,DOS,TMM figure. steps = 1e6
    
    6) Spreading of correlations
   
    
    7) Disorder Model https://journals.aps.org/pre/pdf/10.1103/PhysRevE.76.027701

For all:
    burn_in = 1000
    thinning = 1000

## data/slurm_runs/104690
A repeat of ## data/slurm_runs/103867 but this time with 1e6 mcmc steps

## data/slurm_runs/105461
The above took too long, so I've split it up into 4 x 10 x 1e5 runs for jobs 500 -  599 and will see how this goes
106059 - accidentally repeated 105461
107587 - jobs 600 -  699
108588 - jobs 700 -  799

TODO for this: 
- check is the choice of initial state biased the outcome.
- plot the gap opening with this
- aggregate the data into a nice package and save those?

## 
a repeat of ## data/slurm_runs/103867 but with 1e5 steps and alpha = 3 to show short range behaviour
For the band opening plot for short ranged. It's a fixed J=5, U=1, alpha = 3, with T from 0.1 to 4 with 200 steps
very fine energy spacing: logger = Eigenspectrum_IPR_all(bins = 2000, limit = 5),
Ns = [4,8,16,32,64,128]

N_steps = 1e5,
N_burn_in = 1e3, 
thin = 1000
Repeats = 40

### Successful Jobs in this run:
    data/slurm_runs/108646
    data/slurm_runs/110591
    data/slurm_runs/111591
    data/slurm_runs/112592 SR_gap_open_3-400 2020-01-08T09:56:58.790249 
    data/slurm_runs/112645 SR_gap_open_4-500 2020-01-08T09:58:13.348533 
    data/slurm_runs/112646 SR_gap_open_5-600 2020-01-08T09:59:12.614123 
    data/slurm_runs/112647 SR_gap_open_6-700 2020-01-08T09:59:59.685818 
    data/slurm_runs/112648 SR_gap_open_7-800 2020-01-08T10:00:50.517321 

### Failed:
data/slurm_runs/108647 ### Looks like this was a failure

## 


# These all worked well and I'm using the data in the short paper

## data/slurm_runs/117734 TUphase 2020-01-16T11:03:47.098979 
## data/slurm_runs/117969 TJphase 2020-01-16T15:25:38.281519 
## data/slurm_runs/119596 Talphaphase 2020-01-20T12:17:20.408326 
    data/slurm_runs/128530 Talphaphase2 2020-02-11T14:10:30.403752 A second run of this for averaging
    The above are processed in notebooks/munich/Data_preparation_for_phase_diagrams.ipynb

The below two are processed in notebooks/munich/Data_preparation_for_linear_T_plots.ipynb
## data/slurm_runs/119733-119745 LR_gap_open
    data/slurm_runs/119733 LR_gap_open_0-100 2020-01-20T17:36:03.101930 
    data/slurm_runs/119734 LR_gap_open_1-200 2020-01-20T17:36:42.165883 
    data/slurm_runs/119737 LR_gap_open_2-300 2020-01-20T17:38:06.253792 
    data/slurm_runs/119739 LR_gap_open_3-400 2020-01-20T17:39:19.189426 
    data/slurm_runs/119741 LR_gap_open_4-500 2020-01-20T17:40:23.173205 
    data/slurm_runs/119742 LR_gap_open_5-600 2020-01-20T17:41:15.189140 
    data/slurm_runs/119744 LR_gap_open_6-700 2020-01-20T17:43:51.876766 
    data/slurm_runs/119745 LR_gap_open_7-800 2020-01-20T17:44:32.740390 

## data/slurm_runs/129462 LinearT_biggerU7-800 2020-02-14T16:12:06.643395 
  data/slurm_runs/129518 LinearT_biggerU6-700 2020-02-14T16:13:27.570689 
  data/slurm_runs/129519 LinearT_biggerU5-600 2020-02-17T14:23:06.384628 
  data/slurm_runs/131716 LinearT_biggerU4-500 2020-02-19T19:00:05.486815
  data/slurm_runs/131717 LinearT_biggerU3-400 2020-02-19T19:02:03.622552 
  data/slurm_runs/131718 LinearT_biggerU2-300 2020-02-19T19:04:45.582341 
  data/slurm_runs/131719 LinearT_biggerU1-200 2020-02-19T19:06:19.629829 
  data/slurm_runs/131722 LinearT_biggerU0-100 2020-02-19T19:07:01.709881 

## data/slurm_runs/139591 TUphase 2020-02-27T16:11:22.519995 
    I should check the code to see how this differed from the previous TU phase

## data/slurm_runs/153891 multi_temps 2020-05-05T10:17:26.861660 
    cancelled because these jobs were taking more than a day each

## data/slurm_runs/153974 multi_temps 2020-05-06T16:40:08.780429
there appears to be something weird about this data, all the different Ts look the same
Edit: it was just that all the high T did look the same 

## data/slurm_runs/154077 multitemps_2 2020-05-11T18:46:01.251842 
This one included lower temps

## data/slurm_runs/154101 multitemp_gapped 2020-05-13T12:34:16.997532 
I didn't look at this data

## data/slurm_runs/154259 multiUmultiT 2020-05-18T18:52:45.747042 
I used this in May2020-2.ipynb. The data is a bit noisy.

## data/slurm_runs/154539 multi_temp_U=5 2020-05-23T21:08:08.448137 
Due to bug, this has no data saved

## data/slurm_runs/154619 multi_temp 2020-05-24T22:40:50.598095 
Due to bug, this has no data saved

## data/slurm_runs/154680 multi_T_test_U=5 2020-05-25T15:56:27.516392 

## data/slurm_runs/154806 multi_temp_U=10 2020-05-25T17:03:46.711752 

## data/slurm_runs/155044 mutliT_U=10 2020-05-25T19:21:42.262751 

## data/slurm_runs/155089 CDW_phase 2020-05-25T23:09:17.155136 

## data/slurm_runs/155179 longer_U=10 2020-05-28T12:23:32.987463 

## data/slurm_runs/155289 long_run 2020-06-04T14:50:59.254117 

## data/slurm_runs/155290 test_run 2020-06-07T14:28:14.765969 

## data/slurm_runs/155292 test 2020-06-09T11:02:23.841270 

## data/slurm_runs/155293 test 2020-06-09T11:04:45.647429 

## data/slurm_runs/155294 test 2020-06-09T11:06:08.916613 

## data/slurm_runs/155314 test 2020-06-09T16:54:35.345450 

## data/slurm_runs/155326 test 2020-06-09T16:56:52.343411 

## data/slurm_runs/155338 test 2020-06-09T16:58:07.746951 

## data/slurm_runs/155378 test 2020-06-10T13:27:07.015654 

## data/slurm_runs/155390 test 2020-06-10T13:30:16.530439 

## data/slurm_runs/155402 test 2020-06-10T13:31:24.698995 

## data/slurm_runs/155414 test 2020-06-10T13:43:23.413627 

## data/slurm_runs/155415 test 2020-06-10T13:43:59.902210 

## data/slurm_runs/155416 test 2020-06-10T13:44:17.542055 

## data/slurm_runs/155428 test 2020-06-10T13:47:47.664730 

## data/slurm_runs/155440 test 2020-06-10T13:51:21.858739 

## data/slurm_runs/155476 testest 2020-06-10T14:11:48.327337 

## data/slurm_runs/155488 testtest 2020-06-10T14:14:46.460880 

## data/slurm_runs/155512 testttest 2020-06-10T14:19:12.841483 

## data/slurm_runs/155524 testtest 2020-06-10T14:21:18.268844 
