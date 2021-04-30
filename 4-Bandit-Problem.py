# Importing relevant libraries
import numpy as np
import matplotlib.pyplot as mpy
import random

# Declaring total number of bandit problems
banditProblems=2000
# Declaring total number of arms in each bandit problem
k=4
# Declaring total number of times to pull each arm
armPulls=1000

# Storing the true means generated for each arms for all the bandits
trueMeans=np.random.normal(0,1,(banditProblems,k))
# Storing the true optimal arms in each bandit
trueOptimal=np.argmax(trueMeans,1)

# Storing the array of values for epsilon
epsilon=[0,0.1]
col=['r','g']

# Adding subplots to plot and compare both plots simultaneously
plotFirst=mpy.figure().add_subplot(111)
plotSecond=mpy.figure().add_subplot(111)

# Looping through all the values of epsilon
for x in range(len(epsilon)) :
	
	print('The present epsilon value is : ',x)

	# Storing the predicted reward
	Q=np.zeros((banditProblems,k))
	# Total number of times each arms is pulled
	N=np.ones((banditProblems,k))
	# Assigning the initial random arm pulls
	initialArm=np.random.normal(trueMeans,1)

	# Declaring the variables to store rewards
	rewardEps=[]
	rewardEps.append(0)
	rewardEps.append(np.mean(initialArm))
	rewardEpsOptimal=[]

	for y in range(2,armPulls+1) :
		# Declaring variables to store rewards in the pulls per time-step
		rewardPull=[]
		# Storing the number of pulls of best arm in the present time step
		optimalPull=0
		# Looping through all the bandit problems
		for z in range(banditProblems) :

			if random.random()<epsilon[x] :
				i=np.random.randint(k)
			else :
				i=np.argmax(Q[z])

			# Condition to calculate percentage optimal action
			if i==trueOptimal[z] :
				optimalPull=optimalPull+1

			rewardTemp=np.random.normal(trueMeans[z][i],1)
			rewardPull.append(rewardTemp)
			N[z][i]=N[z][i]+1
			Q[z][i]=Q[z][i]+(rewardTemp-Q[z][i])/N[z][i]

		rewardAvgPull=np.mean(rewardPull)
		rewardEps.append(rewardAvgPull)
		rewardEpsOptimal.append(float(optimalPull)*100/2000)
	plotFirst.plot(range(0,armPulls+1),rewardEps,col[x])
	plotSecond.plot(range(2,armPulls+1),rewardEpsOptimal,col[x])

# Plotting the average reward and optimal reward against 4 arms steps
mpy.rc('text',usetex=True)
plotFirst.title.set_text(r'$\epsilon$-greedy : Average Reward Vs Steps for 4 arms')
plotFirst.set_ylabel('Average Reward')
plotFirst.set_xlabel('Steps')
plotFirst.legend((r"$\epsilon=$"+str(epsilon[0]),r"$\epsilon=$"+str(epsilon[1])),loc='best')
plotSecond.title.set_text(r'$\epsilon$-greedy : $\%$ Optimal Action Vs Steps for 4 arms')
plotSecond.set_ylabel(r'$\%$ Optimal Action')
plotSecond.set_xlabel('Steps')
plotSecond.set_ylim(0,100)
# Adding legend
plotSecond.legend((r"$\epsilon=$"+str(epsilon[0]),r"$\epsilon=$"+str(epsilon[1])),loc='best')
mpy.show()
